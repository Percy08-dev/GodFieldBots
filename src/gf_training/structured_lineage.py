"""Generation2 component chains and retention; no whole-Source qualification.

Generation1/Source/coordinator dispatch remains outside this additive increment.
Historical validation uses only recorded declarations, never the current runtime.
"""
from __future__ import annotations

import hashlib
import math
import os
import secrets
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    closed,
    digest,
    integer,
)
from gf_common.structured import INPUT_RECIPE, action_config_v2
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as GAME_LAYOUT
from gf_common.validation import result as validate_result
from gf_models.config import PPOTrainingConfig, training_config
from gf_models.sharded_arrays import json_bytes
from gf_models.structured_binding import (
    child_cuda_visibility,
    input_binding,
    input_binding_game_v3,
    policy_label,
    recorded_policy_label,
    validate_cuda_visibility,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    ModelConfigV3,
    model_config_v2,
    model_config_v3,
    recorded_architecture_identity,
    recorded_model_config_v2,
)
from gf_models.structured_sealed import (
    CLASSIFICATION,
    SEMANTIC_KEYS,
    SEMANTIC_KEYS_V3,
    SealedScan,
    basename,
    checksum,
    inspect_pruned_game_inference_v3,
    inspect_pruned_inference,
    inspect_retained_artifact,
    inspect_retained_game_artifact_v3,
    pruning_basename,
)
from gf_training.lineage import _OWNED_LOCKS, checkpoint_path, generation_path

RUN_REQUIRED = {'schema_version', 'algorithm', 'run_id', 'status', 'config_sha256', 'effective_config', 'config_inputs', 'identity', 'seed', 'device', 'created_unix', 'last_unix', 'wall_deadline_unix', 'admission_deadline_unix', 'execution_deadline_unix', 'updates', 'accepted_decisions', 'optimizer_steps', 'generation_sha256', 'collection_seconds', 'update_seconds', 'invocations', 'completed_updates', 'model_id', 'model_version', 'input_recipe', 'action_contract', 'purpose', 'synthetic_capacity', 'resume_identity', 'execution'}
RUN_OPTIONAL = {'hardware', 'elapsed_seconds', 'artifact_bytes', 'peak_coordinator_memory_bytes', 'coordinator_memory_scope', 'timing_scope', 'failure', 'quarantine', 'cleanup_failure', 'budget_overrun_seconds'}
GEN_FIELDS = {'schema_version', 'run_id', 'generation', 'parent_sha256', 'model_id', 'model_version', 'input_recipe', 'action_contract', 'purpose', 'synthetic_capacity', 'inference_sha256', 'resume_sha256', 'artifact_files', 'accepted_decisions', 'optimizer_steps', 'update_sha256', 'parent_controls'}
UPDATE_FIELDS = {'schema_version', 'status', 'run_id', 'generation', 'parent_sha256', 'batch_id', 'collection', 'collection_manifest_sha256', 'numeric', 'numeric_sha256', 'accepted_decisions', 'source_proof', 'metrics', 'planned_games', 'actual_games', 'health', 'policy_bindings', 'learner_seats', 'collection_seconds', 'update_seconds', 'model_id', 'model_version', 'input_recipe', 'action_contract', 'purpose', 'synthetic_capacity', 'numeric_files'}
RECEIPT_FIELDS = {'status', 'inference_sha256', 'original_files', 'deletion_intent', 'deletion_receipt'}


class _ChainScan(SealedScan):
    """Carry the same verified parent snapshots through owned operations."""
    def __init__(self, *, directory: Path, max_owned_bytes: int, deadline: float) -> None:
        super().__init__(max_owned_bytes=max_owned_bytes, deadline=deadline)
        self.root = directory
        self.artifact_snapshot: tuple[int, ...] | None = None
        self.artifact_count = 0

    def finalize(self) -> None:
        if self.artifact_snapshot is not None:
            _artifact_directories(self.root, self.artifact_count, self)
        super().finalize()
        if self.artifact_snapshot is not None and _stat_identity((self.root / 'artifacts').lstat()) != self.artifact_snapshot:
            raise ContractError('artifact directory changed during final hash sweep')
        for path, (identity, _) in self.files.items():
            self.check()
            if _stat_identity(path.lstat()) != identity:
                raise ContractError('component post-operation file identity changed')
        self.check()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _filemap(value: Any, required: set[str]) -> Json:
    if not isinstance(value, dict) or not required.issubset(value):
        raise ContractError('required component file map members missing')
    for name, sha in value.items():
        basename(name)
        checksum(sha)
    return value


def _retained_declaration(artifact: Any, state: Json, model: Any) -> None:
    if (canonical(artifact.inference['model']) != canonical(asdict(model)) or artifact.resume is None
            or canonical(artifact.resume['binding']) != canonical(state['execution']['learner']['binding'])
            or artifact.resume['cuda_rng_count'] != (state['hardware']['device_count'] if state['device'] != 'cpu' else 0)):
        raise ContractError('retained model/resources/learner binding/RNG declaration mismatch')


def _retained_values(scan: SealedScan, *values: Any) -> None:
    for value in values:
        scan.charge(32 * json_bytes(value) + 65536)


def _artifact_directories(directory: Path, count: int, scan: SealedScan) -> None:
    scan.charge(1024 * (count + 2), temporary=True)
    expected = {f'{number:06}' for number in range(count + 1)}
    seen = set()
    root = directory / 'artifacts'
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ContractError('component artifact root is a symlink')
    with os.scandir(root) as entries:
        while True:
            scan.check()
            try:
                item = next(entries)
            except StopIteration:
                break
            if item.name not in expected or item.name in seen or not item.is_dir(follow_symlinks=False):
                raise ContractError('missing/foreign component artifact generation directory')
            seen.add(item.name)
    if seen != expected:
        raise ContractError('missing component artifact generation directory')
    if _stat_identity(root.lstat()) != _stat_identity(before):
        raise ContractError('artifact directory changed while enumerating')
    if isinstance(scan, _ChainScan):
        if scan.artifact_snapshot is not None and scan.artifact_snapshot != _stat_identity(before):
            raise ContractError('artifact directory changed since chain validation')
        scan.artifact_snapshot, scan.artifact_count = _stat_identity(before), count
    scan.check()


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ContractError('finite nonnegative lineage number required')
    return float(value)


def _owned(directory: Path) -> Path:
    directory = directory.resolve()
    if directory not in _OWNED_LOCKS:
        raise ContractError('component chain operation requires existing run ownership')
    return directory


def _model(config: Json, recorded_signature: str | None = None, *, _game: bool = False) -> Any:
    raw = config.get('model', {})
    closed(raw, set(ModelConfigV2.__dataclass_fields__) | {'checkpoint', 'inference', 'cuda_visible_devices'})
    if raw['checkpoint'] is not None:
        raise ContractError('component PPO Config must be fresh model2')
    pure = {key: value for key, value in raw.items() if key in ModelConfigV2.__dataclass_fields__}
    if _game:
        return model_config_v3(pure, layout=GAME_LAYOUT)
    return model_config_v2(pure) if recorded_signature is None else recorded_model_config_v2(pure, recorded_signature)


def resume_identity(config_sha256: str, identity: Json, execution: Json, classification: Json) -> str:
    checksum(config_sha256)
    return digest({'recipe': 'structured-ppo-resume-identity-v1', 'config_sha256': config_sha256,
                   'identity': identity, 'execution': execution, 'input_binding': classification})


def _classification(record: Json, state: Json) -> None:
    if (record.get('model_id') != state['model_id'] or record.get('model_version') != state['model_version']
            or record.get('input_recipe') != state['input_recipe']
            or any(canonical(record.get(key)) != canonical(state[key]) for key in (GAME_CLASSIFICATION if state['schema_version'] == 3 else CLASSIFICATION))):
        raise ContractError('component lineage model/input identity mismatch')


def _state(state: Json, *, recorded_signature: str | None = None, _game: bool = False) -> tuple[Any, Any]:
    closed(state, RUN_REQUIRED | (GAME_EXTRA if _game else set()), RUN_OPTIONAL)
    if (type(state['schema_version']) is not int or state['schema_version'] != (3 if _game else 2) or state['algorithm'] != 'ppo'
            or state['status'] not in ('running', 'paused', 'complete') or state['model_version'] != ('3' if _game else '2')
            or state['input_recipe'] != (GAME_LAYOUT.input_recipe if _game else INPUT_RECIPE) or state['purpose'] != ('qualified_game' if _game else 'component') or state['synthetic_capacity'] is not False):
        raise ContractError('healthy component run2 required')
    if (type(state['run_id']) is not str or not state['run_id'] or not isinstance(state['config_inputs'], (list, tuple))
            or not isinstance(state['invocations'], list)):
        raise ContractError('component run identity/invocation declaration invalid')
    config = state['effective_config']
    if not isinstance(config, dict) or digest(config) != state['config_sha256']:
        raise ContractError('component frozen Config hash mismatch')
    actions: Any = action_config_v3(config['actions']) if _game else action_config_v2(config['actions'])
    expected = input_binding_game_v3(actions.action_contract, card_instance_rules()) if _game else input_binding(actions.action_contract, 'component', False)
    if not isinstance(state['identity'], dict) or not state['identity']:
        raise ContractError('recorded component identity object required')
    if any(canonical(state[key]) != canonical(value) for key, value in expected.items()):
        raise ContractError('component run Config classification mismatch')
    if _game:
        from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
        from gf_common.card_instance_runtime import APPROVED_BINARY_SHA256
        from gf_common.preprocessing import preparation_config
        settings = preparation_config(config['preprocessing'])
        preprocessing = {'backend': 'python', 'reader_contract': 'participant-structured-v2',
            'implementation': 'python-reference-player-card-refs-v2', **expected}
        if settings.backend == 'rust':
            from gf_training.native_backend import recorded_card_instance_preprocessing
            preprocessing = state['identity'].get('preprocessing', {})
            recorded_card_instance_preprocessing(preprocessing, binding=expected)
        integration = state['identity'].get('integration', {})
        checksum(integration.get('server_config_sha256'))
        if (settings.backend not in ('python', 'rust') or settings.reader != 'participant-structured-v2'
                or canonical(state['identity'].get('preprocessing')) != canonical(preprocessing)
                or integration.get('binary_sha256') != APPROVED_BINARY_SHA256
                or integration.get('declared_simulator_revision') != actions.action_contract.simulator_revision
                or integration.get('source_revision') != actions.action_contract.simulator_revision
                or canonical(integration.get('source_spec_sha256')) != canonical(dict(CORRECTED_REVIEWED_SPEC_HASHES))):
            raise ContractError('game parent recorded reader/runtime/input identity mismatch')
    model = _model(config, recorded_signature, _game=_game)
    if model.model_id != state['model_id']:
        raise ContractError('component model kind differs from Config')
    training = training_config(config['training'])
    if not isinstance(training, PPOTrainingConfig) or config['run']['split'] != 'train' or state['device'] != training.device:
        raise ContractError('component training role/device mismatch')
    if any(state.get(key) for key in ('failure', 'quarantine', 'cleanup_failure', 'budget_overrun_seconds')):
        raise ContractError('component parent contains failure evidence')
    integer(state['seed'], 0, 2**63 - 1)
    if tuple(training.seeds) != (state['seed'],):
        raise ContractError('component learning seed differs from Config')
    execution = closed(state['execution'], {'learner', 'inference'})
    for role, recorded in execution.items():
        closed(recorded, {'binding', 'cuda_visible_devices'})
        binding = InferenceBindingV2(**closed(recorded['binding'], set(InferenceBindingV2.__dataclass_fields__)))
        device = training.device if role == 'learner' else config['model']['inference']['device']
        visibility = child_cuda_visibility(device, config['model']['cuda_visible_devices'])
        if binding.device != device or validate_cuda_visibility(recorded['cuda_visible_devices']) != visibility:
            raise ContractError('recorded execution/device/visibility mismatch')
    if canonical(execution['inference']['binding']) != canonical(config['model']['inference']):
        raise ContractError('recorded inference execution differs from frozen Config')
    if state['resume_identity'] != resume_identity(state['config_sha256'], state['identity'], execution, expected):
        raise ContractError('domain-separated component resume identity mismatch')
    created, last, wall, admission, execute = [_number(state[key]) for key in ('created_unix', 'last_unix', 'wall_deadline_unix', 'admission_deadline_unix', 'execution_deadline_unix')]
    run = config['run']
    if (not created <= last <= wall or not created < admission <= execute < wall or wall - created > 7200
            or admission - created > 5400 or execute - created > 7140
            or any(abs((limit - created) - run[key]) > .001 for limit, key in ((wall, 'wall_seconds'), (admission, 'admission_seconds'), (execute, 'execution_seconds')))):
        raise ContractError('component original deadline provenance mismatch')
    integer(state['updates'], 0, min(1000, training.max_updates))
    integer(state['accepted_decisions'], 0)
    integer(state['optimizer_steps'], 0, 2**31 - 1)
    if state['completed_updates'] is not None:
        integer(state['completed_updates'], 0, 1000)
    for key in ('collection_seconds', 'update_seconds'):
        _number(state[key])
    if state['status'] in ('paused', 'complete'):
        if abs(_number(state.get('elapsed_seconds')) - (last - created)) > .001:
            raise ContractError('component closing elapsed provenance mismatch')
        if integer(state.get('artifact_bytes'), 0) > training.max_owned_bytes:
            raise ContractError('component original storage limit exceeded')
        for invocation in state['invocations']:
            health = closed(invocation.get('child_health'), {'exit_code', 'stderr_bytes'})
            if type(health['exit_code']) is not int or health['exit_code'] != 0:
                raise ContractError('component learner invocation unhealthy')
            integer(health['stderr_bytes'], 0)
    hardware = closed(state.get('hardware'), {'device', 'name', 'cuda_runtime', 'device_count', 'cuda_driver_api_version', 'exact_driver_build'})
    if (hardware['device'] != state['device'] or type(hardware['name']) is not str or not hardware['name']
            or type(hardware['exact_driver_build']) is not str or not hardware['exact_driver_build']
            or hardware['cuda_runtime'] is not None and type(hardware['cuda_runtime']) is not str):
        raise ContractError('recorded hardware declaration shape mismatch')
    integer(hardware['device_count'], 0, 1024)
    if state['device'] == 'cpu':
        if hardware['device_count'] != 0 or hardware['cuda_driver_api_version'] is not None or hardware['name'] != 'cpu':
            raise ContractError('recorded CPU hardware inconsistent')
    else:
        integer(hardware['cuda_driver_api_version'], 1)
        if hardware['device_count'] == 0 or not hardware['cuda_runtime']:
            raise ContractError('recorded CUDA hardware inconsistent')
    return actions, model


def _controls(value: Json, generation: int) -> None:
    closed(value, {'schedule_opponent_rng', 'snapshot_pool', 'collection_position'})
    if type(value['snapshot_pool']) is not list:
        raise ContractError('component snapshot pool list required')
    for number in value['snapshot_pool']:
        integer(number, 0, generation)
    if value['collection_position'] != generation or type(value['collection_position']) is not int or value['snapshot_pool'] != [number for number in range(generation + 1) if number % 10 == 0][-5:]:
        raise ContractError('component snapshot pool/collection position mismatch')
    rng = closed(value['schedule_opponent_rng'], {'bit_generator', 'state', 'has_uint32', 'uinteger'})
    nested = closed(rng['state'], {'state', 'inc'})
    if rng['bit_generator'] != 'PCG64':
        raise ContractError('component schedule RNG recipe mismatch')
    integer(nested['state'], 0, 2**128 - 1)
    integer(nested['inc'], 1, 2**128 - 1)
    integer(rng['has_uint32'], 0, 1)
    integer(rng['uinteger'], 0, 2**32 - 1)
    if nested['inc'] % 2 != 1:
        raise ContractError('component PCG64 increment invalid')


def _relative(directory: Path, value: Any) -> Path:
    if type(value) is not str or Path(value).is_absolute() or not Path(value).parts or any(part in ('.', '..') for part in Path(value).parts):
        raise ContractError('owned relative lineage path required')
    target = directory / value
    for parent in (target, *target.parents):
        if parent.is_symlink():
            raise ContractError('symlink lineage source path')
        if parent == directory:
            break
    return target


def _update(directory: Path, update: Json, state: Json, previous: Json, scan: SealedScan, model: Any,
            inference_declaration: Json, *, _game: bool = False) -> int:
    classification = GAME_CLASSIFICATION if _game else CLASSIFICATION
    closed(update, UPDATE_FIELDS | (GAME_EXTRA if _game else set()))
    integer(update['generation'], 1, 1000)
    for key in ('parent_sha256', 'collection_manifest_sha256', 'numeric_sha256'):
        checksum(update[key])
    proof = update['source_proof']
    if not isinstance(proof, dict):
        raise ContractError('component Source proof object required')
    checksum(proof.get('manifest_sha256'))
    _filemap(update['numeric_files'], {'rollout.json'})
    if update['numeric_files']['rollout.json'] != update['numeric_sha256']:
        raise ContractError('numeric required manifest hash mismatch')
    _classification(update, state)
    if (type(update['schema_version']) is not int or update['schema_version'] != (3 if _game else 2) or update['status'] != 'complete'
            or update['run_id'] != state['run_id'] or update['generation'] != previous['generation'] + 1
            or update['parent_sha256'] != scan.files[generation_path(directory, previous['generation'])][1]
            or type(update['planned_games']) is not int or update['planned_games'] != 16 or update['actual_games'] != 16
            or update['health'] != {}):
        raise ContractError('component update2 links/games/health invalid')
    seats = update['learner_seats']
    if not isinstance(seats, dict) or len(seats) != 16:
        raise ContractError('all16 original learner roles required')
    for role in seats.values():
        integer(role, 1, 2)
    collection = _relative(directory, update['collection'])
    manifest = scan.object(collection / 'manifest.json', update['collection_manifest_sha256'])
    if any(canonical(manifest.get(key)) != canonical(state[key]) for key in classification):
        raise ContractError('component collection classification mismatch')
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != (5 if _game else 4)
            or type(manifest.get('journal_schema_version')) is not int or manifest['journal_schema_version'] != (6 if _game else 5)):
        raise ContractError('component update requires collection4/journal5')
    numeric = _relative(directory, update['numeric'])
    rollout = scan.object(numeric / 'rollout.json', update['numeric_sha256'])
    if (type(rollout.get('schema_version')) is not int or rollout['schema_version'] != (4 if _game else 3) or rollout.get('kind') != 'structured-rollout'
            or rollout.get('architecture_identity') != recorded_architecture_identity(model, **({"layout": GAME_LAYOUT} if _game else {}))
            or rollout.get('action_config_hash') != digest(state['effective_config']['actions'])
            or any(canonical(rollout.get(key)) != canonical(state[key]) for key in classification)):
        raise ContractError('component numeric3 model/input links mismatch')
    if any(canonical(rollout.get(key)) != canonical(inference_declaration[key]) for key in (SEMANTIC_KEYS_V3 if type(model) is ModelConfigV3 else SEMANTIC_KEYS)):
        raise ContractError('numeric recorded semantic declarations differ from inference')
    if _game:
        from gf_models.structured_binding import policy_label_game_v3
        expected_label = policy_label_game_v3(model, InferenceBindingV2(**state['execution']['inference']['binding']), previous['inference_sha256'],
            action_contract=action_config_v3(state['effective_config']['actions']).action_contract, rules=card_instance_rules(),
            action_config_hash=digest(state['effective_config']['actions']), cuda_visible_devices=state['execution']['inference']['cuda_visible_devices'])
    else:
        expected_label = recorded_policy_label(model, InferenceBindingV2(**state['execution']['inference']['binding']), previous['inference_sha256'],
                                 cuda_visible_devices=state['execution']['inference']['cuda_visible_devices'])
    if rollout.get('policy_version') != expected_label:
        raise ContractError('numeric frozen learner inference identity mismatch')
    episodes = rollout.get('episodes')
    if (not isinstance(episodes, list) or len(episodes) != 16
            or any(episode.get('end_kind') != 'natural' or len(episode.get('frames', [])) != len(episode.get('selections', [])) for episode in episodes)):
        raise ContractError('numeric must preserve every original natural learner episode')
    count = integer(rollout.get('decision_count'), 1)
    owners = set()
    for episode in episodes:
        integer(episode.get('actor'), 1, 2)
        owner = (episode['actor'], episode.get('episode'))
        if type(owner[1]) is not str or not owner[1] or owner in owners:
            raise ContractError('numeric episode ownership duplicated or invalid')
        owners.add(owner)
        validate_result(episode.get('outcome'))
        if (episode.get('outcome') is None or episode.get('reason') is not None
                or episode.get('has_targets') is not bool(episode['frames'])
                or any(canonical(episode.get(key)) != canonical(state[key]) for key in classification)):
            raise ContractError('numeric natural episode/classification/target declaration mismatch')
        if any(any(canonical(frame.get(key)) != canonical(state[key]) for key in classification) for frame in episode['frames']):
            raise ContractError('numeric frame classification mismatch')
    if count != sum(len(episode['frames']) for episode in episodes) or update['accepted_decisions'] != count or type(update['accepted_decisions']) is not int:
        raise ContractError('learner-only numeric decision counter mismatch')
    proof = update['source_proof']
    if (type(proof.get('schema_version')) is not int or proof['schema_version'] != 1
            or proof.get('kind') != ('card-instance-original-source-proof' if _game else 'structured-component-source-proof') or proof.get('qualified') is not False
            or proof.get('parent_qualification') != 'not_performed' or proof.get('games') != 16
            or proof.get('manifest_sha256') != update['collection_manifest_sha256']):
        raise ContractError('component Source proof scalar links mismatch')
    if any(canonical(proof.get(key)) != canonical(state[key]) for key in classification):
        raise ContractError('component Source proof classification mismatch')
    checksum(proof.get('source_set_sha256'))
    if integer(proof.get('accepted_decisions'), 0) < count:
        raise ContractError('both-view Source count below selected learner count')
    expected_files = {**rollout['data']['files'], 'rollout.json': update['numeric_sha256']}
    if canonical(expected_files) != canonical(update['numeric_files']):
        raise ContractError('numeric exact original file map mismatch')
    scan.inventory(numeric, set(expected_files))
    scan.bundle(numeric, rollout['data'])
    _number(update['collection_seconds'])
    _number(update['update_seconds'])
    return count


def validate_component_chain(directory: Path, *, require_complete: bool = True, require_retained: bool = False,
                             max_owned_bytes: int, deadline: float) -> tuple[Json, list[Json]]:
    directory = directory.resolve()
    scan = _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes, deadline=deadline)
    return _validate_component_chain(directory, require_complete=require_complete, require_retained=require_retained, scan=scan)


def _validate_component_chain(directory: Path, *, require_complete: bool, require_retained: bool, scan: _ChainScan, _game: bool = False) -> tuple[Json, list[Json]]:
    state = scan.object(directory / 'run.json')
    # Descriptor bootstrap is bound to the original protected generation0 and
    # its mandatory manifest map/hash before a missing event field is interpreted.
    bootstrap = closed(scan.object(generation_path(directory, 0)), GEN_FIELDS | (GAME_EXTRA if _game else set()))
    integer(bootstrap['generation'], 0, 0)
    _filemap(bootstrap['artifact_files'], {'inference.json', 'resume.json'})
    checksum(bootstrap['inference_sha256'])
    if bootstrap['artifact_files']['inference.json'] != bootstrap['inference_sha256']:
        raise ContractError('generation0 bootstrap inference/map mismatch')
    first_inference = scan.object(checkpoint_path(directory, 0) / 'inference.json', bootstrap['inference_sha256'])
    actions, model = _state(state, recorded_signature=first_inference['token_signature'], _game=_game)
    if canonical(first_inference['model']) != canonical(asdict(model)):
        raise ContractError('generation0 bootstrap model differs from original frozen Config')
    if require_complete and (state['status'] != 'complete' or state['updates'] == 0 or state['completed_updates'] != state['updates']):
        raise ContractError('complete nonzero component chain required')
    latest = closed(scan.object(directory / 'latest.json'), {'generation', 'sha256'})
    checksum(latest['sha256'])
    checksum(state['generation_sha256'])
    integer(latest['generation'], 0, 1000)
    count = state['updates']
    _artifact_directories(directory, count, scan)
    if latest['generation'] != count or latest['sha256'] != state['generation_sha256']:
        raise ContractError('component run/latest counters mismatch')
    scan.inventory(directory / 'generations', {f'{number:06}.json' for number in range(count + 1)})
    scan.inventory(directory / 'updates', {f'{number:06}.json' for number in range(1, count + 1)})
    registry = closed(scan.object(directory / 'artifact-registry.json'), {'schema_version', 'generations'})
    if (type(registry['schema_version']) is not int or registry['schema_version'] != (3 if _game else 2)
            or not isinstance(registry['generations'], dict) or set(registry['generations']) != {str(number) for number in range(count + 1)}):
        raise ContractError('component artifact registry2 inventory mismatch')
    records: list[Json] = []
    parent, accepted, steps = None, 0, 0
    for number in range(count + 1):
        record = closed(scan.object(generation_path(directory, number)), GEN_FIELDS | (GAME_EXTRA if _game else set()))
        integer(record['accepted_decisions'], 0)
        integer(record['optimizer_steps'], 0, 2**31 - 1)
        checksum(record['inference_sha256'])
        checksum(record['resume_sha256'])
        _filemap(record['artifact_files'], {'inference.json', 'resume.json'})
        if (record['artifact_files']['inference.json'] != record['inference_sha256']
                or record['artifact_files']['resume.json'] != record['resume_sha256']):
            raise ContractError('required generation manifest map hashes mismatch')
        if number:
            checksum(record['parent_sha256'])
            checksum(record['update_sha256'])
        elif record['parent_sha256'] is not None or record['update_sha256'] is not None:
            raise ContractError('generation0 nullable links must be null')
        _classification(record, state)
        if (type(record['schema_version']) is not int or record['schema_version'] != (3 if _game else 2) or record['run_id'] != state['run_id']
                or type(record['generation']) is not int or record['generation'] != number or record['parent_sha256'] != parent):
            raise ContractError('component generation2 hash chain mismatch')
        _controls(record['parent_controls'], number)
        receipt = closed(registry['generations'][str(number)], RECEIPT_FIELDS)
        if canonical(receipt['original_files']) != canonical(record['artifact_files']):
            raise ContractError('component original artifact map differs from registry')
        path = checkpoint_path(directory, number)
        retained = number == 0 or number % 10 == 0 or number == count
        if receipt['status'] == 'retained':
            if receipt['deletion_intent'] is not None or receipt['deletion_receipt'] is not None:
                raise ContractError('retained artifact has deletion evidence')
            artifact = _inspect_pair(path, actions=actions, game=_game, expected_inference_digest=record['inference_sha256'],
                scan=scan)
            _retained_declaration(artifact, state, model)
            if (canonical(artifact.files) != canonical(record['artifact_files']) or artifact.files['resume.json'] != record['resume_sha256']
                    or artifact.resume is None or artifact.resume['optimizer_steps'] != record['optimizer_steps']
                    or artifact.resume['resume_identity'] != state['resume_identity']):
                raise ContractError('component retained resume/map/counter mismatch')
        elif not retained and not require_retained and receipt['status'] == 'deleted':
            artifact = _inspect_pruned(path, actions=actions, game=_game, expected_inference_digest=record['inference_sha256'], expected_resume_digest=record['resume_sha256'],
                original_files=record['artifact_files'], deletion_receipt=receipt, scan=scan)
        else:
            raise ContractError('required retained generation or completed pruning receipt missing')
        if (artifact.inference['architecture_identity'] != recorded_architecture_identity(model, **({"layout": GAME_LAYOUT} if _game else {}))
                or artifact.inference['action_config_hash'] != digest(state['effective_config']['actions'])
                or receipt['inference_sha256'] != record['inference_sha256']):
            raise ContractError('component artifact/model/action identity mismatch')
        if number:
            update_path = directory / 'updates' / f'{number:06}.json'
            update = scan.object(update_path, record['update_sha256'])
            amount = _update(directory, update, state, records[-1], scan, model, artifact.inference, _game=_game)
            accepted += amount
            expected_steps = steps + 4 * math.ceil(amount / 128)
            if record['accepted_decisions'] != accepted or record['optimizer_steps'] != expected_steps or update['metrics'].get('optimizer_steps') != expected_steps:
                raise ContractError('component generation learner/Adam counters mismatch')
        elif record['accepted_decisions'] != 0 or record['optimizer_steps'] != 0 or record['update_sha256'] is not None:
            raise ContractError('component generation0 counters invalid')
        steps = record['optimizer_steps']
        parent = scan.files[generation_path(directory, number)][1]
        records.append(record)
    if parent != latest['sha256'] or accepted != state['accepted_decisions'] or steps != state['optimizer_steps']:
        raise ContractError('component final generation/run counters mismatch')
    _artifact_directories(directory, count, scan)
    scan.finalize()
    return state, records


def _write(path: Path, value: Json, *, scan: SealedScan, immutable: bool = False) -> str:
    if not isinstance(scan, _ChainScan):
        raise ContractError('owned lineage writer requires chain snapshots')
    if not path.is_relative_to(scan.root):
        raise ContractError('foreign lineage publication path')
    for directory in (path.parent, *path.parent.parents):
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ContractError('lineage publication parent must be owned nonsymlink directory')
        if directory == scan.root:
            break
    scan.charge(32 * json_bytes(value) + 65536)
    raw = (canonical(value) + '\n').encode()
    scan.check()
    temporary = path.with_name(path.name + '.tmp')
    if path.is_symlink():
        raise ContractError('refusing mutable lineage symlink')
    fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w+b') as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
        scan.check()
        before = os.fstat(target.fileno())
        if immutable:
            os.link(temporary, path, follow_symlinks=False)
        else:
            os.replace(temporary, path)
        actual = path.lstat()
        if (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) != (before.st_dev, before.st_ino, len(raw), before.st_mtime_ns):
            raise ContractError('published lineage object differs from owned temporary')
        target.seek(0)
        if target.read(len(raw) + 1) != raw:
            raise ContractError('published lineage content differs from owned bytes')
        after = os.fstat(target.fileno())
        current = path.lstat()
        if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                != (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns)
                or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
                != (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns)):
            raise ContractError('published lineage identity changed during verification')
        if immutable:
            temporary.unlink()
        scan.check()
        post_unlink = os.fstat(target.fileno())
        if ((post_unlink.st_dev, post_unlink.st_ino, post_unlink.st_size, post_unlink.st_mtime_ns)
                != (before.st_dev, before.st_ino, len(raw), before.st_mtime_ns)):
            raise ContractError('owned lineage object changed after temporary unlink')
        # Unlink legitimately changes ctime/nlink. Capture that SAME held object,
        # then verify the final path and bytes against this post-unlink baseline.
        published_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(published_fd, 'rb') as published:
            if _stat_identity(os.fstat(published.fileno())) != _stat_identity(post_unlink):
                raise ContractError('published lineage path replaced after temporary unlink')
            if published.read(len(raw) + 1) != raw:
                raise ContractError('published lineage bytes changed after temporary unlink')
            if (_stat_identity(os.fstat(published.fileno())) != _stat_identity(post_unlink)
                    or _stat_identity(os.fstat(target.fileno())) != _stat_identity(post_unlink)
                    or _stat_identity(path.lstat()) != _stat_identity(post_unlink)):
                raise ContractError('published lineage path/object changed after verification')
        scan.check()
        # A verified write by this owner advances, rather than discards, its
        # existing snapshot. Other parent records remain bound to their reads.
        scan.files[path] = (_stat_identity(post_unlink), hashlib.sha256(raw).hexdigest())
    return hashlib.sha256(raw).hexdigest()


def commit_component_generation(directory: Path, state: Json, *, generation: int, update: Json | None, controls: Json,
                                artifact_files: Json, max_owned_bytes: int, deadline: float, _game: bool = False,
                                _precommit: Any = None) -> Json:
    directory = _owned(directory)
    actions, model = _state(state, _game=_game)
    assert type(model) is (ModelConfigV3 if _game else ModelConfigV2)
    if state['status'] != 'running':
        raise ContractError('only a running component parent can append a generation')
    integer(generation, 0, 1000)
    _controls(controls, generation)
    scan = _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes, deadline=deadline)
    _retained_values(scan, state, controls, update, artifact_files)
    _filemap(artifact_files, {'inference.json', 'resume.json'})
    path = checkpoint_path(directory, generation)
    artifact = _inspect_pair(path, actions=actions, game=_game, expected_inference_digest=checksum(artifact_files.get('inference.json')),
        scan=scan)
    _retained_declaration(artifact, state, model)
    if (canonical(artifact.files) != canonical(artifact_files) or artifact.resume is None
            or artifact.resume['resume_identity'] != state['resume_identity'] or artifact.resume['optimizer_steps'] != state['optimizer_steps']):
        raise ContractError('component commit artifact/resume identity mismatch')
    parent = state['generation_sha256']
    if generation:
        if generation != state['updates'] + 1 or update is None:
            raise ContractError('component commit must append exactly one update')
        previous = closed(scan.object(generation_path(directory, generation - 1), parent), GEN_FIELDS | (GAME_EXTRA if _game else set()))
        _classification(previous, state)
        durable = scan.object(directory / 'run.json')
        stable_fields = (RUN_REQUIRED | (GAME_EXTRA if _game else set())) - {'accepted_decisions', 'optimizer_steps', 'collection_seconds', 'update_seconds', 'last_unix', 'invocations', 'completed_updates'}
        if any(canonical(durable.get(key)) != canonical(state[key]) for key in stable_fields):
            raise ContractError('component commit differs from durable frozen parent')
        amount = _update(directory, update, state, previous, scan, model, artifact.inference, _game=_game)
        if state['accepted_decisions'] != previous['accepted_decisions'] + amount or state['optimizer_steps'] != previous['optimizer_steps'] + 4 * math.ceil(amount / 128):
            raise ContractError('component commit counters mismatch')
        registry = scan.object(directory / 'artifact-registry.json')
    else:
        if update is not None or parent is not None or state['updates'] != 0 or state['accepted_decisions'] != 0 or state['optimizer_steps'] != 0:
            raise ContractError('component initial commit counters mismatch')
        registry = {'schema_version': 3 if _game else 2, 'generations': {}}
    scan.finalize()
    if _precommit is not None:
        _precommit.prepare_commit()
    update_sha = _write(directory / 'updates' / f'{generation:06}.json', update, scan=scan, immutable=True) if update is not None else None
    record = {'schema_version': 3 if _game else 2, 'run_id': state['run_id'], 'generation': generation, 'parent_sha256': parent,
        **{key: state[key] for key in ('model_id', 'model_version', 'input_recipe', *(GAME_CLASSIFICATION if _game else CLASSIFICATION))},
        'inference_sha256': artifact_files['inference.json'], 'resume_sha256': artifact_files['resume.json'],
        'artifact_files': dict(artifact_files), 'accepted_decisions': state['accepted_decisions'], 'optimizer_steps': state['optimizer_steps'],
        'update_sha256': update_sha, 'parent_controls': controls}
    sha = _write(generation_path(directory, generation), record, scan=scan, immutable=True)
    registry['generations'][str(generation)] = {'status': 'retained', 'inference_sha256': record['inference_sha256'],
        'original_files': dict(artifact_files), 'deletion_intent': None, 'deletion_receipt': None}
    _write(directory / 'artifact-registry.json', registry, scan=scan)
    _write(directory / 'latest.json', {'generation': generation, 'sha256': sha}, scan=scan)
    state.update(updates=generation, generation_sha256=sha)
    _write(directory / 'run.json', state, scan=scan)
    return record


def retain_component_generations(directory: Path, generation: int, *, max_owned_bytes: int, deadline: float, _game: bool = False) -> None:
    directory = _owned(directory)
    integer(generation, 0, 1000)
    predecessor = generation - 1
    if predecessor <= 0 or predecessor % 10 == 0:
        return
    scan = _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes, deadline=deadline)
    state, records = _validate_component_chain(directory, require_complete=False, require_retained=False, scan=scan, _game=_game)
    if state['updates'] != generation:
        raise ContractError('retention requires the current committed generation')
    registry = scan.object(directory / 'artifact-registry.json')
    record = records[predecessor]
    receipt = registry['generations'][str(predecessor)]
    if receipt['status'] != 'retained':
        raise ContractError('retention requires full original artifacts')
    path = checkpoint_path(directory, predecessor)
    actions = action_config_v3(state['effective_config']['actions']) if _game else action_config_v2(state['effective_config']['actions'])
    _inspect_pair(path, expected_inference_digest=record['inference_sha256'], actions=actions, game=_game, scan=scan)
    targets = {name: sha for name, sha in record['artifact_files'].items() if name != 'inference.json'}
    scan.finalize()
    _prune_staged(directory, path, registry, receipt, targets, record['inference_sha256'], scan)


def _directory_entry(parent_fd: int, name: str, held_fd: int) -> None:
    actual = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    held = os.fstat(held_fd)
    if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (held.st_dev, held.st_ino):
        raise ContractError('owned pruning directory entry was replaced')


def _fd_hash(fd: int, size: int, scan: SealedScan) -> str:
    value, offset = hashlib.sha256(), 0
    while offset < size:
        scan.check()
        block = os.pread(fd, min(1024**2, size - offset), offset)
        if not block:
            raise ContractError('pruning file shortened while verifying')
        value.update(block)
        offset += len(block)
    if os.pread(fd, 1, size):
        raise ContractError('pruning file grew while verifying')
    scan.check()
    return value.hexdigest()


def _prune_parent(parent: Path, parent_fd: int, original_identity: tuple[int, int], scan: _ChainScan,
                  *, advance: bool = False) -> None:
    held, current = os.fstat(parent_fd), parent.lstat()
    if (not stat.S_ISDIR(current.st_mode) or (held.st_dev, held.st_ino) != original_identity
            or (current.st_dev, current.st_ino) != original_identity
            or _stat_identity(held) != _stat_identity(current)):
        raise ContractError('pruning artifacts parent differs from original held identity')
    if not advance and _stat_identity(held) != scan.artifact_snapshot:
        raise ContractError('pruning artifacts parent changed since verified snapshot')
    if advance:
        scan.artifact_snapshot = _stat_identity(held)
    scan.check()


def _prune_staged(directory: Path, path: Path, registry: Json, receipt: Json, targets: Json,
                  inference_sha: str, scan: _ChainScan) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = directory / 'artifacts'
    staging = pruning_basename('prune-' + secrets.token_hex(16))
    if scan.artifact_snapshot is None:
        raise ContractError('pruning requires original artifacts directory snapshot')
    original_parent = (scan.artifact_snapshot[0], scan.artifact_snapshot[1])
    parent_fd = os.open(parent, flags)
    source_fd = stage_fd = None
    phase, target_name = 'create_stage', None
    try:
        _prune_parent(parent, parent_fd, original_parent, scan)
        source_fd = os.open(path.name, flags, dir_fd=parent_fd)
        _directory_entry(parent_fd, path.name, source_fd)
        os.mkdir(staging, 0o700, dir_fd=parent_fd)  # exclusive; never adopt a collision.
        _prune_parent(parent, parent_fd, original_parent, scan, advance=True)
        stage_fd = os.open(staging, flags, dir_fd=parent_fd)
        if stat.S_IMODE(os.fstat(stage_fd).st_mode) != 0o700:
            raise ContractError('pruning staging must have private0700 mode')
        _directory_entry(parent_fd, staging, stage_fd)
        receipt.update(status='deleting', deletion_intent={'files': targets, 'staging_basename': staging})
        _write(directory / 'artifact-registry.json', registry, scan=scan)
        for name, expected in targets.items():
            scan.check()
            _prune_parent(parent, parent_fd, original_parent, scan)
            target_name, phase = basename(name), 'verify_public'
            _directory_entry(parent_fd, path.name, source_fd)
            _directory_entry(parent_fd, staging, stage_fd)
            held = None
            try:
                with scan.file(path / name, expected) as (original, before):
                    held = os.dup(original.fileno())
                try:
                    os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ContractError('private staged target already exists')
                phase = 'rename_to_stage'
                os.rename(name, name, src_dir_fd=source_fd, dst_dir_fd=stage_fd)
                phase = 'verify_moved'
                _directory_entry(parent_fd, staging, stage_fd)
                moved = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
                current = os.fstat(held)
                if (not stat.S_ISREG(moved.st_mode) or (moved.st_dev, moved.st_ino) != (before.st_dev, before.st_ino)
                        or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                        or _stat_identity(moved) != _stat_identity(current)):
                    raise ContractError('moved pruning object differs from verified original')
                if _fd_hash(held, current.st_size, scan) != expected:
                    raise ContractError('moved pruning content differs from verified original')
                if (_stat_identity(os.fstat(held)) != _stat_identity(moved)
                        or _stat_identity(os.stat(name, dir_fd=stage_fd, follow_symlinks=False)) != _stat_identity(moved)):
                    raise ContractError('staged pruning object changed during verification')
                phase = 'unlink_private'
                _directory_entry(parent_fd, staging, stage_fd)
                os.unlink(name, dir_fd=stage_fd)  # never unlink an untrusted public artifact basename.
                scan.files.pop(path / name)
            finally:
                if held is not None:
                    os.close(held)
        phase, target_name = 'verify_completion', None
        _directory_entry(parent_fd, path.name, source_fd)
        _directory_entry(parent_fd, staging, stage_fd)
        for fd, expected_names in ((source_fd, {'inference.json'}), (stage_fd, set())):
            seen = set()
            scan.charge(4096, temporary=True)
            with os.scandir(fd) as entries:
                while True:
                    scan.check()
                    try:
                        entry = next(entries)
                    except StopIteration:
                        break
                    if entry.name not in expected_names or entry.name in seen or not entry.is_file(follow_symlinks=False):
                        raise ContractError('pruning completion inventory has replacement/foreign entry')
                    seen.add(entry.name)
            if seen != expected_names:
                raise ContractError('pruning completion inventory missing inference')
        inf_fd = os.open('inference.json', os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=source_fd)
        try:
            info = os.fstat(inf_fd)
            if (_stat_identity(info) != scan.files[path / 'inference.json'][0]
                    or _fd_hash(inf_fd, info.st_size, scan) != inference_sha
                    or _stat_identity(os.fstat(inf_fd)) != _stat_identity(info)):
                raise ContractError('pruning preserved inference changed')
        finally:
            os.close(inf_fd)
        _directory_entry(parent_fd, staging, stage_fd)
        _directory_entry(parent_fd, path.name, source_fd)
        _prune_parent(parent, parent_fd, original_parent, scan)
        os.rmdir(staging, dir_fd=parent_fd)  # only this verified empty owned directory.
        _prune_parent(parent, parent_fd, original_parent, scan, advance=True)
        scan.inventories[path] = ('inference.json',)
        # Accept only this owner's known stage creation/removal; enumerate the
        # original generation directories again, then recheck all parent files.
        _artifact_directories(directory, scan.artifact_count, scan)
        scan.finalize()
        receipt.update(status='deleted', deletion_receipt={'files': targets})
        _write(directory / 'artifact-registry.json', registry, scan=scan)
    except BaseException as error:
        # Preserve captured foreign objects and incomplete intent; no restoration
        # or recursive cleanup can overwrite a public replacement.
        if stage_fd is not None:
            try:
                fd = os.open('failure.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=stage_fd)
                with os.fdopen(fd, 'w') as output:
                    output.write(canonical({'phase': phase, 'target': target_name, 'failure': type(error).__name__}) + '\n')
            except BaseException as diagnostic_error:  # noqa: BLE001 - initiating failure survives diagnostic failure
                error.add_note('pruning failure diagnostic: ' + type(diagnostic_error).__name__)
        raise
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)


def owned_component_binding(directory: Path, generation: int, *, max_owned_bytes: int, deadline: float) -> tuple[str, RuntimePolicyBinding]:
    directory = _owned(directory)
    scan = _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes, deadline=deadline)
    state, records = _validate_component_chain(directory, require_complete=False, require_retained=False, scan=scan)
    integer(generation, 0, len(records) - 1)
    actions, model = _state(state)
    assert type(model) is ModelConfigV2
    record = records[generation]
    registry = scan.object(directory / 'artifact-registry.json')
    if registry['generations'][str(generation)]['status'] != 'retained':
        raise ContractError('owned component binding cannot load a pruned generation')
    inspect_retained_artifact(checkpoint_path(directory, generation), expected_inference_digest=record['inference_sha256'],
        action_contract=actions.action_contract, purpose='component', synthetic_capacity=False, scan=scan)
    execution = state['execution']['inference']
    binding = InferenceBindingV2(**execution['binding'])
    label = policy_label(model, binding, record['inference_sha256'], cuda_visible_devices=execution['cuda_visible_devices'])
    options = {'version': '2', 'checkpoint': str(checkpoint_path(directory, generation)), 'expected_manifest_digest': record['inference_sha256'],
        'model': asdict(model), 'inference': asdict(binding), 'cuda_visible_devices': execution['cuda_visible_devices'],
        **input_binding(actions.action_contract, 'component', False)}
    scan.finalize()
    return label, RuntimePolicyBinding(model.model_id, options)


GAME_EXTRA = {'rule_signature', 'layout_signature'}
GAME_CLASSIFICATION = CLASSIFICATION | GAME_EXTRA


def _inspect_pair(directory: Path, *, expected_inference_digest: str, actions: Any,
                  game: bool, scan: SealedScan) -> Any:
    if game:
        return inspect_retained_game_artifact_v3(directory, expected_inference_digest=expected_inference_digest,
            actions=actions, rules=card_instance_rules(), scan=scan)
    return inspect_retained_artifact(directory, expected_inference_digest=expected_inference_digest,
        action_contract=actions.action_contract, purpose='component', synthetic_capacity=False, scan=scan)


def _inspect_pruned(directory: Path, *, actions: Any, game: bool, **options: Any) -> Any:
    if game:
        return inspect_pruned_game_inference_v3(directory, actions=actions, rules=card_instance_rules(), **options)
    return inspect_pruned_inference(directory, action_contract=actions.action_contract,
        purpose='component', synthetic_capacity=False, **options)


def validate_game_chain(directory: Path, *, require_complete: bool = True, require_retained: bool = False,
                        max_owned_bytes: int, deadline: float) -> tuple[Json, list[Json]]:
    directory = directory.resolve()
    scan = _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes, deadline=deadline)
    return _validate_component_chain(directory, require_complete=require_complete,
        require_retained=require_retained, scan=scan, _game=True)


def commit_game_generation(directory: Path, state: Json, *, projection: Any = None, **options: Any) -> Json:
    if options.get('generation'):
        from gf_training.card_instance_source import IssuedGameProjection
        if type(projection) is not IssuedGameProjection or projection.source.state['run_id'] != state['run_id']:
            raise ContractError('game generation requires its live Source-issued update')
        projection.require(projection.process, after=True)
    return commit_component_generation(directory, state, _game=True, _precommit=projection, **options)


def retain_game_generations(directory: Path, generation: int, *, max_owned_bytes: int, deadline: float) -> None:
    retain_component_generations(directory, generation, max_owned_bytes=max_owned_bytes, deadline=deadline, _game=True)
