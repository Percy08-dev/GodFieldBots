"""Original corrected-game Source ownership and explicitly scoped PPO projection.

Serialized evidence never substitutes for the same live lock/process/lease owner.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.card_instance_runtime import validate_runtime_receipt
from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    closed,
    digest,
    integer,
)
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.schedule import serialize_schedule
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature
from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT, validate_neural_metadata
from gf_models.structured_binding import (
    input_binding_game_v3,
    policy_label_game_v3,
    safe_options_game_v3,
)
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_sealed import SEMANTIC_KEYS_V3, SealedScan, checksum
from gf_training.collection import CollectionPlan, training_groups
from gf_training.lineage import _OWNED_LOCKS, checkpoint_path
from gf_training.structured_lineage import _ChainScan, _state, _validate_component_chain
from gf_training.structured_rollout import (
    _episode_correspondence_digest,
    _owned_bytes,
    _predicted_token_bytes,
    _source_correspondence,
    _write_rollout,
    game_rollout,
    read_game_rollout,
)
from gf_training.structured_source import (
    _inspect_component_collection,
    _name,
    _SourceOwner,
    _SourceSnapshot,
)

LINKAGE_KEYS = {'parent_run_id', 'parent_run_path', 'batch_id', 'collection_generation',
                'learner_seats', 'schedule_digest', 'policy_registry_digest'}
PROJECTION_KEYS = {'recipe', 'parent_run_id', 'batch_id', 'collection_generation', 'expected_optimizer_steps',
                   'expected_policy_version', 'collection_manifest_sha256', 'numeric_manifest_sha256',
                   'numeric_files_sha256', 'source_correspondence'}


def frozen_linkage(state: Json, directory: Path, generation: int, groups: Any, registry: Any) -> Json:
    return {'parent_run_id': state['run_id'], 'parent_run_path': str(directory),
        'batch_id': f'batch-{generation:06}', 'collection_generation': generation,
        'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes},
        'schedule_digest': digest(serialize_schedule(groups, 4)),
        'policy_registry_digest': digest({key: asdict(value) for key, value in registry.items()})}


def _game_manifest(owner: _SourceOwner, manifest: Json, actions: ActionConfigV3,
                   expected_split: str | None) -> tuple[Json, list[tuple[Json, Json]]]:
    expected = input_binding_game_v3(actions.action_contract, card_instance_rules())
    if (manifest.get('schema_version') != 5 or type(manifest['schema_version']) is not int
            or manifest.get('status') != 'complete' or manifest.get('collection_role') != 'ppo-batch'
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
    link = closed(manifest.get('linkage'), LINKAGE_KEYS)
    integer(link['collection_generation'], 0, 999)
    if link['batch_id'] != f"batch-{link['collection_generation']:06}" or not Path(link['parent_run_path']).is_absolute():
        raise ContractError('original PPO batch linkage mismatch')
    specs = [(group, spec) for group in manifest.get('schedule', []) for spec in group.get('episodes', [])]
    names = [_name(spec.get('episode_id')) for _, spec in specs]
    episodes = manifest.get('episodes')
    if (len(specs) != 16 or len(set(names)) != 16 or type(episodes) is not list or len(episodes) != 16
            or {row.get('episode_id') for row in episodes} != set(names)
            or len({row.get('game_id') for row in episodes}) != 16
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
    expected_names = set()
    for entry in files:
        closed(entry, {'path', 'sha256', 'bytes'})
        path = Path(entry['path'])
        if path.is_absolute() or any(part in ('.', '..') for part in path.parts) or not path.parts:
            raise ContractError('foreign original inventory path')
        name = path.as_posix()
        if name in expected_names or name in ('manifest.json', 'report.json'):
            raise ContractError('duplicate/foreign original inventory entry')
        expected_names.add(name)
        # Opaque diagnostics are owned/hashes only; never become learner input.
        owner.read(owner.root / path)
        if owner.files[name]['sha256'] != checksum(entry['sha256']) or owner.files[name]['bytes'] != entry['bytes']:
            raise ContractError('original collection inventory hash/size mismatch')
    actual = owner.inventory('**/*')
    if set(actual) != expected_names | {'manifest.json', 'report.json'}:
        raise ContractError('complete original collection inventory changed')
    return {row['episode_id']: row for row in episodes}, specs


def _game_header(header: Json, manifest: Json, episode: Json, spec: Json,
                 actions: ActionConfigV3, viewer: int) -> None:
    if (header.get('run_id') != manifest['run_id'] or header.get('episode_id') != spec['episode_id']
            or header.get('viewer') != viewer or header.get('game_id') != episode['game_id']
            or header.get('policy') != spec['policies'][viewer - 1]
            or header.get('sampling_seed') != spec['sampling_seeds'][viewer - 1]):
        raise ContractError('original game participant identity/stream mismatch')
    binding = closed(manifest['policy_bindings'][header['policy']], {'model_id', 'policy_options'})
    if header['model_id'] != binding['model_id']:
        raise ContractError('original game model identity mismatch')
    index = next(index for index, group in enumerate(manifest['schedule']) if any(row['episode_id'] == spec['episode_id'] for row in group['episodes']))
    if canonical(header.get('runtime_evidence')) != canonical(manifest['runtime_evidence'][index]):
        raise ContractError('original participant runtime receipt mismatch')
    if binding['policy_options'] is None:
        if header['model_version'] != '1' or header['policy'] != binding['model_id'] + '-card-instance-v2':
            raise ContractError('original baseline identity mismatch')
    else:
        raw = manifest['neural_inputs']['policies'][header['policy']]
        safe, label = validate_neural_metadata(raw, actions=actions, rules=card_instance_rules())
        if header['model_version'] != '3' or label != header['policy'] or canonical(safe) != canonical(binding['policy_options']) or canonical(header.get('neural')) != canonical(raw):
            raise ContractError('original neural metadata differs from frozen registry')


def _binding(directory: Path, state: Json, record: Json, generation: int) -> tuple[str, RuntimePolicyBinding]:
    actions, model = _state(state, _game=True)
    execution = state['execution']['inference']
    binding = InferenceBindingV2(**execution['binding'])
    label = policy_label_game_v3(model, binding, record['inference_sha256'], action_contract=actions.action_contract,
        rules=card_instance_rules(), action_config_hash=digest(asdict(actions)), cuda_visible_devices=execution['cuda_visible_devices'])
    options = safe_options_game_v3({'version': '3', 'checkpoint': str(checkpoint_path(directory, generation)),
        'expected_manifest_digest': record['inference_sha256'], 'model': asdict(model), 'inference': asdict(binding),
        'cuda_visible_devices': execution['cuda_visible_devices'], **input_binding_game_v3(actions.action_contract, card_instance_rules()),
        'action_config_hash': digest(asdict(actions))}, actions=actions, rules=card_instance_rules())
    return label, RuntimePolicyBinding(model.model_id, options)


def recorded_plan(directory: Path, state: Json, records: list[Json], generation: int,
                  deadline: float) -> tuple[CollectionPlan, str, Json]:
    import numpy as np
    label, learner = _binding(directory, state, records[generation], generation)
    registry = {'random-card-instance-v2': RuntimePolicyBinding('random'),
        'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic'), label: learner}
    controls = records[generation]['parent_controls']
    snapshots = []
    for number in controls['snapshot_pool']:
        key, value = _binding(directory, state, records[number], number)
        registry[key] = value
        snapshots.append(key)
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = controls['schedule_opponent_rng']
    groups = training_groups(state['seed'], generation, label, tuple(snapshots), rng, actions_version=3)
    link = frozen_linkage(state, directory, generation, groups, registry)
    return CollectionPlan(groups, registry, directory / 'collections' / link['batch_id'], link, deadline, deadline), label, dict(rng.bit_generator.state)


@dataclass(eq=False)
class _OnlineGameOwner:
    ownership: Any
    state: Json
    process: Any
    ledger: Any
    active: bool = True
    issued: Any = None
    collection_controls: Any = None
    collection_generation: int | None = None

    def check(self, *, steps: int | None = None) -> None:
        child, ledger = self.process, self.ledger
        lease = ledger.learner
        if (not self.active or _OWNED_LOCKS.get(self.ownership.directory) != self.ownership.token
                or child.directory.resolve() != self.ownership.directory or lease is None or child.lease is not lease
                or lease._cleanup_owner is not child or child.ledger is not ledger or lease.released or lease.creation_pending
                or child.process is None or child.process.returncode is not None or lease.pid != child.process.pid
                or child.steps != (self.state['optimizer_steps'] if steps is None else steps)
                or canonical(child.hardware) != canonical(self.state['hardware'])):
            raise ContractError('SAME live game parent lock/learner/process/lease/ledger required')


def validate_owned_collection(plan: CollectionPlan, ledger: Any, *, max_owned_bytes: int,
                              deadline: float) -> _OnlineGameOwner:
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.structured_learning_process import GameLearningProcess
    if type(ledger) is not ReservationLedger or ledger.learner is None or ledger.workers or ledger.updating:
        raise ContractError('online collection requires idle SAME learner ledger')
    child = ledger.learner._cleanup_owner
    if type(child) is not GameLearningProcess or type(getattr(child, 'online_owner', None)) is not _OnlineGameOwner:
        raise ContractError('online collection requires concrete wire4 live owner')
    owner = child.online_owner
    owner.check()
    generation = plan.linkage.get('collection_generation')
    if owner.collection_controls is None:
        guard = SealedScan(max_owned_bytes=max_owned_bytes, deadline=deadline)
        frozen = owner.state['effective_config']['server']['config']
        with guard.file(Path(frozen).absolute(), owner.state['identity']['integration']['server_config_sha256']):
            pass
        owner.collection_controls, owner.collection_generation = guard, generation
    elif owner.collection_generation != generation:
        raise ContractError('previous original collection controls have not retired')
    owner.collection_controls.finalize()
    state, records = validate_game_parent(owner.ownership.directory, require_complete=False,
        max_owned_bytes=max_owned_bytes, deadline=deadline)
    actions, model = _state(state, _game=True)
    from gf_training.structured_numeric_worker import wire_identity
    expected_identity = wire_identity(model, InferenceBindingV2(**state['execution']['learner']['binding']),
        digest(asdict(actions)), state['resume_identity'], action_contract=actions.action_contract,
        purpose='qualified_game', synthetic_capacity=False,
        cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'], _actions=actions)
    if (canonical(child.identity) != canonical(expected_identity) or canonical(asdict(child.model)) != canonical(asdict(model))
            or canonical(asdict(child.binding)) != canonical(state['execution']['learner']['binding'])
            or child.visibility != state['execution']['learner']['cuda_visible_devices']
            or ledger.learner_bytes != 0 or ledger.learner.resident_bytes != child.resident_bytes):
        raise ContractError('online learner wire/model/execution/reservation differs from original parent')
    expected, _, _ = recorded_plan(owner.ownership.directory, state, records, state['updates'], deadline)
    if (state['status'] != 'running' or set(state) != set(owner.state)
            or any(canonical(value) != canonical(owner.state.get(key)) for key, value in state.items() if key != 'last_unix')
            or type(owner.state.get('last_unix')) not in (int, float) or not math.isfinite(owner.state['last_unix'])
            or owner.state['last_unix'] < state['last_unix']
            or plan.output.absolute() != expected.output or canonical(plan.linkage) != canonical(expected.linkage)
            or canonical(serialize_schedule(plan.groups, 4)) != canonical(serialize_schedule(expected.groups, 4))
            or canonical({key: asdict(value) for key, value in plan.bindings.items()}) != canonical({key: asdict(value) for key, value in expected.bindings.items()})
            or plan.kind != 'ppo_batch' or ledger.global_cuda_bytes != state['effective_config']['run']['global_cuda_budget_bytes']
            or ledger.max_workers != state['effective_config']['run']['max_participant_workers']):
        raise ContractError('online collection differs from frozen original parent/control/schedule/registry')
    return owner


def _check_collection_parent(manifest: Json, state: Json, plan: CollectionPlan) -> None:
    if (canonical(manifest['config']) != canonical(state['effective_config']) or manifest['config_hash'] != state['config_sha256']
            or canonical(manifest.get('inputs')) != canonical(state['config_inputs'])
            or canonical(manifest.get('code')) != canonical(state['identity']['code'])
            or canonical(manifest['linkage']) != canonical(plan.linkage)
            or canonical(manifest['schedule']) != canonical(serialize_schedule(plan.groups, 4))
            or canonical(manifest['policy_bindings']) != canonical({key: asdict(value) for key, value in plan.bindings.items()})):
        raise ContractError('original game collection differs from frozen parent/plan')
    integration = state['identity']['integration']
    for receipt in manifest['runtime_evidence']:
        if receipt['binary']['sha256'] != integration['binary_sha256']:
            raise ContractError('original collection executed binary differs from parent')


@dataclass(eq=False)
class IssuedGameProjection:
    """Private-lifecycle update handle; only its live Source can issue permission."""
    source: Any = field(repr=False)
    process: Any = field(repr=False)
    _expectation: bytes = field(repr=False)
    numeric: Path
    files: Json
    active: bool = True
    completed: bool = False
    prepared: bool = False

    @property
    def expectation(self) -> Json:
        return json.loads(self._expectation)

    def require(self, process: Any, *, after: bool = False) -> None:
        if (type(self.source) is not _GameBatchSource or type(self.source.online) is not _OnlineGameOwner
                or not self.active or self.source.issued is not self or self.process is not process or self.source.online.issued is not self
                or self.source.online.state['updates'] != self.expectation['collection_generation']):
            raise ContractError('update requires the SAME active Source-issued projection')
        before = self.expectation['expected_optimizer_steps']
        expected_after = before + 4 * ((self.source.decision_count + 127) // 128)
        self.source.online.check(steps=expected_after if after else before)
        if not after and (self.completed or process.steps != before):
            raise ContractError('replayed/stale Source update transaction')
        if not after:
            self.source.recheck_originals()
            self.source.controls.finalize()
            self.source.online.collection_controls.finalize()

    def prepare_commit(self) -> None:
        self.require(self.process, after=True)
        if not self.completed or self.prepared:
            raise ContractError('Source update has not completed exactly once')
        self.source.recheck_originals()
        self.source.online.collection_controls.finalize()
        controls = self.source.controls
        from gf_training.structured_lineage import _artifact_directories
        previous = controls.artifact_snapshot
        current = (controls.root / 'artifacts').lstat()
        if previous is None or (current.st_dev, current.st_ino) != previous[:2]:
            raise ContractError('owned update replaced original artifacts root')
        controls.artifact_snapshot = None
        _artifact_directories(controls.root, self.expectation['collection_generation'] + 1, controls)
        controls.finalize()
        if self.source.native_inputs is not None:
            self.source.native_inputs.prepare_commit()
        # No remaining reportable per-batch I/O. Old run/registry snapshots are
        # legitimately replaced by the ensuing owned lineage commit.
        self.prepared = True


@dataclass(eq=False)
class _GameBatchSource:
    snapshot: _SourceSnapshot | None
    controls: _ChainScan
    state: Json
    plan: CollectionPlan
    model: Any
    actions: ActionConfigV3
    declaration: Json
    learner_label: str
    online: _OnlineGameOwner | None
    records: tuple[Any, ...] | None
    native_inputs: Any = field(default=None, repr=False)
    issued: IssuedGameProjection | None = None
    _control_base: int = field(init=False)
    _numeric: tuple[Path, str, bytes] | None = None

    def __post_init__(self) -> None:
        from gf_training.native_card_instance import _NativeGameInputs
        if self.native_inputs is None:
            if type(self.snapshot) is not _SourceSnapshot or type(self.records) is not tuple:
                raise ContractError('Python game Source requires its concrete snapshot/records')
        elif (type(self.native_inputs) is not _NativeGameInputs or self.snapshot is not None or self.records is not None):
            raise ContractError('native game inputs are a mutually exclusive concrete representation')
        self._control_base = self.controls.retained

    @property
    def decision_count(self) -> int:
        if self.native_inputs is not None:
            return int(self.native_inputs.decision_count)
        assert self.records is not None
        return sum(len(record.frames) for record in self.records)

    @property
    def episode_digests(self) -> tuple[str, ...]:
        if self.native_inputs is not None:
            return tuple(self.native_inputs.correspondence['episode_digests'])
        assert self.records is not None
        return tuple(_episode_correspondence_digest(record) for record in self.records)

    def recheck_originals(self) -> None:
        if self.native_inputs is not None:
            self.native_inputs.caller_growth(self.controls.retained - self._control_base)
            self.native_inputs.recheck()
        else:
            assert self.snapshot is not None
            self.snapshot.recheck()

    @property
    def proof(self) -> Json:
        if self.native_inputs is not None:
            return self.native_inputs.proof
        assert self.snapshot is not None
        return self.snapshot.collection.proof

    def correspondence(self) -> Json:
        from gf_training.lineage import generation_path
        generation = self.plan.linkage['collection_generation']
        immutable = {str(path.relative_to(self.controls.root)): self.controls.files[path][1] for path in (
            generation_path(self.controls.root, generation), checkpoint_path(self.controls.root, generation) / 'inference.json')}
        fingerprint = digest({'run_id': self.state['run_id'], 'config_sha256': self.state['config_sha256'],
            'identity': self.state['identity'], 'execution': self.state['execution'], 'plan': self.plan.linkage, 'files': immutable,
            'source_set_sha256': self.proof['source_set_sha256']})
        expected = _source_correspondence(fingerprint, self.declaration, self.episode_digests, game=True)
        if self.native_inputs is not None and canonical(expected) != canonical(self.native_inputs.correspondence):
            raise ContractError('native correspondence differs from SAME Source control fingerprint')
        return expected

    def write_numeric(self, directory: Path) -> Json:
        if self.native_inputs is not None:
            self.native_inputs.caller_growth(self.controls.retained - self._control_base)
            return self.native_inputs.publish_numeric(directory)
        assert self.snapshot is not None and self.records is not None
        extra = self.snapshot.extra_bytes_for(self.records)
        self.snapshot.reserve(self.snapshot.additional_bytes((self.records, self.model)) + _predicted_token_bytes(self.records))
        rollout = game_rollout(self.records, self.model, actions=self.actions, rules=card_instance_rules(), reserved_bytes=extra)
        return _write_rollout(directory, rollout, max_owned_bytes=self.snapshot.max_owned_bytes,
            extra_owned_bytes=self.snapshot.extra_bytes_for((rollout.model, rollout.episodes, rollout.locations,
                rollout.action_config_hash, rollout.policy_version, rollout.input_binding)),
            check=self.snapshot.check, reserve=self.snapshot.reserve)

    def read_numeric(self, directory: Path, expected_digest: str) -> Any:
        if self.native_inputs is not None:
            self.native_inputs.caller_growth(self.controls.retained - self._control_base)
            rollout = self.native_inputs.read_numeric(directory, expected_digest)
            files = self.native_inputs.files_for(directory, expected_digest)
            self._numeric = (directory, expected_digest, canonical(files).encode())
            return rollout
        assert self.snapshot is not None
        snapshot = self.snapshot
        prior = snapshot._external_bytes
        self.snapshot.reserve(prior + 32 * (directory / 'rollout.json').lstat().st_size + 65536)
        manifest = self.controls.object(directory / 'rollout.json', expected_digest)
        growth = self.controls.retained - self._control_base
        self.snapshot.reserve(prior + growth)
        rollout = read_game_rollout(directory, expected_digest, self.model, actions=self.actions, rules=card_instance_rules(),
            declaration=self.declaration, expected_correspondence=self.correspondence(),
            max_owned_bytes=self.snapshot.max_owned_bytes, extra_owned_bytes=self.snapshot.resident_bytes + growth,
            check=self.snapshot.check, reserve=lambda peak: snapshot.reserve(peak + growth))
        files = {'rollout.json': expected_digest, **manifest['data']['files']}
        for name, expected in files.items():
            with self.controls.file(directory / name, expected):
                pass
        self.snapshot.reserve(self.snapshot._external_bytes + self.controls.retained - self._control_base - growth)
        self._numeric = (directory, expected_digest, canonical(files).encode())
        return rollout

    def issue(self, numeric: Path, files: Json) -> IssuedGameProjection:
        if self.online is None or self.issued is not None or self.online.issued is not None:
            raise ContractError('only one live online Source may issue an update')
        self.online.check()
        expected_numeric = self.online.ownership.directory / 'numeric' / self.plan.linkage['batch_id']
        if numeric.absolute() != expected_numeric:
            raise ContractError('Source update numeric directory differs from its owned parent/batch')
        if self._numeric is None or self._numeric[:2] != (numeric, files.get('rollout.json')):
            rollout = self.read_numeric(numeric, checksum(files.get('rollout.json')))
            rollout.require_learning()
            del rollout
        if self._numeric is None or canonical(files).encode() != self._numeric[2] or self.decision_count == 0:
            raise ContractError('Source update requires its verified complete nonempty numeric projection/files')
        self.recheck_originals()
        self.controls.finalize()
        self.online.collection_controls.finalize()
        generation = self.plan.linkage['collection_generation']
        expected = {'recipe': 'card-instance-source-update-v1', 'parent_run_id': self.state['run_id'],
            'batch_id': self.plan.linkage['batch_id'], 'collection_generation': generation,
            'expected_optimizer_steps': self.state['optimizer_steps'], 'expected_policy_version': self.learner_label,
            'collection_manifest_sha256': self.proof['manifest_sha256'], 'numeric_manifest_sha256': files['rollout.json'],
            'numeric_files_sha256': digest(dict(sorted(files.items()))), 'source_correspondence': self.correspondence()}
        self.issued = IssuedGameProjection(self, self.online.process, canonical(expected).encode(), numeric, files)
        self.online.issued = self.issued
        if self.native_inputs is not None:
            self.native_inputs.begin_update()
        return self.issued


@contextmanager
def _open_batch(plan: CollectionPlan, state: Json, records: list[Json], controls: _ChainScan,
                settings: PreparationConfig, *, online: _OnlineGameOwner | None,
                max_owned_bytes: int, deadline: float, extra_owned_bytes: int = 0) -> Iterator[_GameBatchSource]:
    with ExitStack() as scope:
        actions, model = _state(state, _game=True)
        generation = plan.linkage['collection_generation']
        expected, label, _ = recorded_plan(controls.root, state, records, generation, deadline)
        if canonical(plan.linkage) != canonical(expected.linkage):
            raise ContractError('original batch plan differs from committed generation')
        previous = controls.object(checkpoint_path(controls.root, generation) / 'inference.json', records[generation]['inference_sha256'])
        native_inputs = None
        if settings.backend == 'rust':
            from gf_training.native_card_instance import _NativeGameInputs
            mode = 'live_cohort' if online is not None else 'historical_parent'
            native_plan = {'schema_version': 3, 'mode': mode,
                'parents': [{'directory': str(controls.root), 'require_complete': False}],
                'collections': ([{'directory': str(plan.output), 'parent_index': 0}] if online is not None else [])}
            if online is not None:
                native_plan['cohort_collection_index'] = 0
            from gf_training.ppo_run import owned_bytes
            remaining_storage = max(0, state['effective_config']['training']['max_owned_bytes'] - owned_bytes(controls.root, deadline))
            native_inputs = scope.enter_context(_NativeGameInputs(native_plan, settings, actions=actions,
                max_owned_bytes=max_owned_bytes, deadline=deadline, extra_owned_bytes=controls.retained + extra_owned_bytes,
                max_scratch_bytes=remaining_storage))
            assert native_inputs.summary is not None
            native_inputs.collection_index = next((index for index, row in enumerate(native_inputs.summary['collections'])
                if row['generation'] == generation), -1)
            if (canonical(native_inputs.collection['linkage']) != canonical(plan.linkage)
                    or native_inputs.collection['learner_label'] != label
                    or canonical(asdict(native_inputs.models[0])) != canonical(asdict(model))
                    or canonical(native_inputs.collection['semantics']) != canonical({key: previous[key] for key in SEMANTIC_KEYS_V3})):
                raise ContractError('native originals differ from SAME recorded Source plan/model/declaration')
            source = _GameBatchSource(None, controls, state, plan, model, actions,
                {key: previous[key] for key in SEMANTIC_KEYS_V3}, label, online, None, native_inputs=native_inputs)
            source.correspondence()
        else:
            original = _SourceOwner(plan.output, settings, deadline, extra_owned_bytes=controls.retained + extra_owned_bytes,
                                    max_owned_bytes=max_owned_bytes)
            collection = _inspect_component_collection(original, actions, 'train', _game=True)
            snapshot = _SourceSnapshot(collection, original, _owned_bytes((collection.episodes, collection._manifest, collection._proof, original.files, original.inventories)))
            _check_collection_parent(collection.manifest, state, plan)
            seats = plan.linkage['learner_seats']
            selected = tuple(row for row in collection.episodes if seats[row.episode_id] == row.viewer)
            if len(selected) != 16 or any(row.model_id != model.model_id or row.model_version != '3' for row in selected):
                raise ContractError('all16 original neural learner seats required')
            source = _GameBatchSource(snapshot, controls, state, plan, model, actions,
                {key: previous[key] for key in SEMANTIC_KEYS_V3}, label, online, tuple(row.records for row in selected))
        try:
            yield source
            if source.issued is None or not source.issued.prepared:
                source.recheck_originals()
                controls.finalize()
                if online is not None and online.collection_controls is not None:
                    online.collection_controls.finalize()
        except BaseException as error:
            if source.issued is None or not source.issued.prepared:
                try:
                    source.recheck_originals()
                    controls.finalize()
                    if online is not None and online.collection_controls is not None:
                        online.collection_controls.finalize()
                except BaseException as cleanup_error:  # noqa: BLE001 - preserve initiating failure
                    error.add_note('game Source final recheck: ' + str(cleanup_error))
            raise
        finally:
            if source.issued is not None:
                source.issued.active = False
                if online is not None:
                    online.issued = None
                    if source.issued.prepared:
                        online.collection_controls = None
                        online.collection_generation = None


@contextmanager
def game_batch_source(plan: CollectionPlan, ledger: Any, *, max_owned_bytes: int, deadline: float,
                      extra_owned_bytes: int = 0) -> Iterator[_GameBatchSource]:
    online = validate_owned_collection(plan, ledger, max_owned_bytes=max_owned_bytes, deadline=deadline)
    controls = _ChainScan(directory=online.ownership.directory, max_owned_bytes=max_owned_bytes, deadline=deadline)
    state, records = _validate_component_chain(online.ownership.directory, require_complete=False,
        require_retained=False, scan=controls, _game=True)
    settings = preparation_config(state['effective_config']['preprocessing'])
    with _open_batch(plan, state, records, controls, settings, online=online,
        max_owned_bytes=max_owned_bytes, deadline=deadline, extra_owned_bytes=extra_owned_bytes) as source:
        yield source


def validate_game_parent(directory: Path, *, require_complete: bool = True, max_owned_bytes: int,
                         deadline: float, extra_owned_bytes: int = 0) -> tuple[Json, list[Json]]:
    directory = directory.resolve()
    scan = _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes - extra_owned_bytes, deadline=deadline)
    state, records = _validate_component_chain(directory, require_complete=require_complete,
        require_retained=False, scan=scan, _game=True)
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(state['seed']))
    if canonical(records[0]['parent_controls']) != canonical({'schedule_opponent_rng': rng.bit_generator.state,
            'snapshot_pool': [0], 'collection_position': 0}):
        raise ContractError('original generation0 PCG64 differs from frozen seed')
    settings = preparation_config(state['effective_config']['preprocessing'])
    if settings.backend == 'rust':
        from gf_training.native_card_instance import _NativeGameInputs
        native_plan = {'schema_version': 3, 'mode': 'historical_parent',
            'parents': [{'directory': str(directory), 'require_complete': require_complete}], 'collections': []}
        with _NativeGameInputs(native_plan, settings, actions=_state(state, _game=True)[0],
                max_owned_bytes=max_owned_bytes, deadline=deadline, extra_owned_bytes=scan.retained + extra_owned_bytes) as inputs:
            assert inputs.summary is not None
            parent = inputs.summary['parent']
            if (parent['identity_sha256'] != digest(state['identity'])
                    or parent['run_sha256'] != scan.files[directory / 'run.json'][1]
                    or any(canonical(state[key]) != canonical(value) for key, value in parent['state'].items())):
                raise ContractError('native historical parent differs from original game control state')
            scan.finalize()
        return state, records
    source_inventories = []
    for number in range(1, len(records)):
        plan, _, after = recorded_plan(directory, state, records, number - 1, deadline)
        if canonical(after) != canonical(records[number]['parent_controls']['schedule_opponent_rng']):
            raise ContractError('original schedule RNG advance mismatch')
        update = scan.object(directory / 'updates' / f'{number:06}.json', records[number]['update_sha256'])
        with _open_batch(plan, state, records, scan, settings, online=None, max_owned_bytes=max_owned_bytes,
                         deadline=deadline, extra_owned_bytes=extra_owned_bytes) as source:
            if canonical(source.proof) != canonical(update['source_proof']):
                raise ContractError('original Source proof differs from committed generation')
            rollout = source.read_numeric(directory / update['numeric'], update['numeric_sha256'])
            rollout.require_learning()
            if len(rollout) != update['accepted_decisions']:
                raise ContractError('original numeric decision count mismatch')
            del rollout
            assert source.snapshot is not None
            original = source.snapshot._owner
            for name, entry in original.files.items():
                scan.charge(1024 + 4 * len(name))
                path = original.root / name
                snapshot = (tuple(entry['identity']), entry['sha256'])
                if path in scan.files and scan.files[path] != snapshot:
                    raise ContractError('original Source changed across parent/batch snapshots')
                scan.files[path] = snapshot
            scan.charge(_owned_bytes(original.inventories) + 512 + 4 * len(str(original.root)))
            source_inventories.append((original.root, tuple(original.inventories)))
    def inventories() -> None:
        for root, patterns in source_inventories:
            comparison = _SourceOwner(root, settings, deadline, extra_owned_bytes=scan.retained + extra_owned_bytes, max_owned_bytes=max_owned_bytes)
            for pattern, names in patterns:
                if comparison._inventory_names(pattern, retain=False) != names:
                    raise ContractError('original Source inventory changed across batches')
    inventories()
    scan.finalize()
    inventories()
    from gf_training.structured_lineage import _stat_identity
    for path, (identity, _) in scan.files.items():
        scan.check()
        if _stat_identity(path.lstat()) != identity:
            raise ContractError('original Source changed after final inventory sweep')
    return state, records
