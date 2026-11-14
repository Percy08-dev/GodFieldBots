"""Original corrected-game Source ownership and explicitly scoped PPO projection.

Serialized evidence never substitutes for the same live lock/process/lease owner.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from gf_common.card_instance import ActionConfigV3
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
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label
from gf_common.schedule import serialize_schedule
from gf_models.card_instance_artifacts import validate_neural_metadata
from gf_models.structured_binding import (
    input_binding_game_v3,
    policy_label_game_v3,
    safe_options_game_v3,
)
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_sealed import SEMANTIC_KEYS_V3, SealedScan, checksum
from gf_training.card_instance_collection_source import inspect_game_manifest as _game_manifest  # noqa: F401 - retained private compatibility alias
from gf_training.collection import CollectionPlan, training_groups
from gf_training.lineage import _OWNED_LOCKS, checkpoint_path
from gf_training.structured_lineage import _ChainScan, _state, _validate_component_chain
from gf_training.structured_rollout import (
    EpisodeRecords,
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
    _SourceOwner,
    _SourceSnapshot,
)

LINKAGE_KEYS = {'parent_run_id', 'parent_run_path', 'batch_id', 'collection_generation',
                'learner_seats', 'schedule_digest', 'policy_registry_digest'}


def _hpo(state: Json) -> bool:
    from gf_training.hpo_source import is_hpo
    return is_hpo(state)


def _packed(state: Json) -> bool:
    return state.get('schema_version') == 4 or _hpo(state) and 'storage_binding' in state


@dataclass(frozen=True)
class OriginalGameDecision:
    frame: Any
    bank: Any
    table: Any
    selection: Any


class OriginalGameEpisodeBorrow:
    """Factory-bound original immutable DTO lease; no public envelope authority."""
    _parent: Any
    _source: Any
    _loan: Any
    _record: EpisodeRecords | None
    _packet: Any
    _read_owner: Any
    _episode_id: str
    _original_identity: tuple[int, str, str]
    _resident: int
    _export: int
    _closed: bool
    _scope_retired: bool
    _iterating: bool
    _failure: BaseException | None
    _iterator: _OriginalDecisionIterator | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ContractError('original episode borrows are returned only by actual-owner factories')

    @classmethod
    def _from_owner(cls, parent: Any, source: Any, budget_loan: Any, *, episode_id: str,
                 record: EpisodeRecords, packet: Any = None, read_owner: Any = None) -> OriginalGameEpisodeBorrow:
        self=object.__new__(cls)
        from gf_training.card_instance_formal import FinalSourceBudgetLoan, FormalGameParent
        if (type(parent) is not FormalGameParent or type(budget_loan) is not FinalSourceBudgetLoan
                or parent._unit_borrow is not None or type(record) is not EpisodeRecords):
            raise ContractError('actual parent, concrete resource loan and original DTO required')
        budget_loan.require(parent)
        if source is not budget_loan._source or source is None:
            raise ContractError('SAME retained Source/custodian loan required for episode borrowing')
        self._parent, self._source, self._loan = parent, source, budget_loan
        self._record = record
        self._packet, self._read_owner = packet, read_owner
        self._episode_id = episode_id
        self._original_identity = (record.bank.actor, record.bank.episode,
            _episode_correspondence_digest(record))
        shared=source._selected is not None and any(record is row for row in source._selected)
        self._resident = 1024 if shared else _owned_bytes(record) + 1024
        self._export = 0 if packet is None else sum(value.nbytes for banks in packet[1:] for value in banks)
        self._closed = self._iterating = self._scope_retired = False
        self._failure = None
        self._iterator = None
        parent._unit_borrow = self
        return self

    def recheck(self) -> None:
        if self._failure is not None:
            raise ContractError('original episode borrow is permanently failed') from self._failure
        if self._closed or self._parent._unit_borrow is not self:
            raise ContractError('SAME active original episode borrow required')
        try:
            self._loan.require(self._parent)
            self._source.recheck()
            if self._read_owner is not None:
                self._read_owner.finalize()
        except BaseException as error:
            self._failure = error
            self._parent._poison(error)
            raise

    @property
    def episode_id(self) -> str:
        self.recheck()
        assert self._record is not None
        identity = (self._record.bank.actor, self._record.bank.episode,
            _episode_correspondence_digest(self._record))
        if identity != self._original_identity:
            error = ContractError('original episode bank/digest identity changed')
            self._failure = error
            self._parent._poison(error)
            raise error
        return self._episode_id

    @property
    def records(self) -> EpisodeRecords:
        self.recheck()
        assert self._record is not None
        return self._record

    @property
    def viewer(self) -> int:
        return self.records.bank.actor

    @property
    def end_kind(self) -> str:
        return self.records.end_kind

    @property
    def outcome(self) -> Any:
        self._loan.admit_local(self._source._reserved_bytes+self._resident+65536)
        return self.records.outcome

    @property
    def decision_count(self) -> int:
        return len(self.records.frames)

    @property
    def episode_digest(self) -> str:
        return _episode_correspondence_digest(self.records)

    @property
    def resident_bytes(self) -> int:
        self.recheck()
        return self._resident

    @property
    def export_bytes(self) -> int:
        self.recheck()
        return self._export

    def iter_decisions(self) -> Iterator[OriginalGameDecision]:
        self.recheck()
        if self._iterator is not None:
            raise ContractError('original decision iterators cannot overlap')
        iterator=_OriginalDecisionIterator(self)
        self._iterator=iterator
        return iterator

    @property
    def cleanup_complete(self) -> bool:
        return self._closed and self._scope_retired and self._record is None and self._packet is None and self._read_owner is None

    def close(self) -> None:
        if self._closed:
            return
        failure = self._failure
        try:
            if failure is None:
                self.recheck()
        except BaseException as error:  # noqa: BLE001 - cleanup retains the initiating original guard failure
            failure = failure or error
        if self._iterator is not None:
            self._iterator.close()
        self._record = self._packet = self._read_owner = None
        self._resident = self._export = 0
        if self._parent._unit_borrow is not self:
            failure = failure or ContractError('actual original unit-borrow exclusion changed')
        else:
            self._closed = True
        if failure is not None:
            self._failure = failure
            vars(failure)['_original_game_episode_cleanup_owner'] = self
            raise failure


    def _finish_scope(self) -> None:
        # Factory's control/record locals have retired before exclusion release.
        if not self._closed or self._record is not None or self._packet is not None:
            raise ContractError('actual original borrow graphs must retire before scope release')
        if self._parent._unit_borrow is not self:
            raise ContractError('SAME original borrow scope required')
        self._parent._unit_borrow=None
        self._scope_retired=True


class _OriginalDecisionIterator:
    def __init__(self, borrow: OriginalGameEpisodeBorrow) -> None:
        self._borrow: OriginalGameEpisodeBorrow | None=borrow
        self._index=0

    def __iter__(self) -> _OriginalDecisionIterator:
        return self

    def __next__(self) -> OriginalGameDecision:
        borrow=self._borrow
        if borrow is None:
            raise StopIteration
        borrow.recheck()
        record=borrow.records
        if self._index>=len(record.frames):
            self.close()
            borrow.recheck()
            raise StopIteration
        borrow._loan.admit_local(borrow._source._reserved_bytes+borrow._resident+_owned_bytes(record.bank)+65536)
        frame,table,selection=record.frames[self._index],record.tables[self._index],record.selections[self._index]
        self._index+=1
        return OriginalGameDecision(frame,record.bank.prefix(frame.boundary),table,selection)

    def close(self) -> None:
        if self._borrow is not None:
            self._borrow._iterator=None
            self._borrow=None


@contextmanager
def _borrow_original_training_episode(parent: Any, generation: int, ordinal: int, *, budget_loan: Any) -> Iterator[OriginalGameEpisodeBorrow]:
    from gf_common.domain import integer
    from gf_common.preprocessing import PreparationConfig
    from gf_training.card_instance_formal import FinalSourceBudgetLoan, FormalGameParent
    if type(parent) is not FormalGameParent or type(budget_loan) is not FinalSourceBudgetLoan:
        raise ContractError('actual formal parent/resource loan required')
    budget_loan.require(parent)
    source = budget_loan._source
    if source is None or not source._active or parent._unit_borrow is not None:
        raise ContractError('SAME retained final Source and idle unit borrow required')
    parent.recheck()
    sizes = parent.canonical_control_sizes
    base = source._reserved_bytes
    budget_loan.admit_local(base+8*(32*sum(sizes)+65536))
    state, records = parent.state, parent.records
    assert state is not None and records is not None
    integer(generation,0,state['updates']-1)
    integer(ordinal,0,15)
    plan, _, _ = recorded_plan(parent.directory,state,records,generation,min(parent.deadline,budget_loan.deadline))
    specs = tuple(spec for group in plan.groups for spec in group.episodes)
    spec = specs[ordinal]
    actions, model = _state(state,_game=True)
    raw_settings = closed(state['effective_config']['preprocessing'],set(PreparationConfig.__dataclass_fields__))
    settings = preparation_config(raw_settings)
    if canonical(asdict(settings)) != canonical(raw_settings):
        raise ContractError('original settings must roundtrip without defaults')
    packet = owner = borrow = None
    record: Any = None
    primary: BaseException | None = None
    try:
        if settings.backend == 'rust':
            native = parent._original_history_native_owner()
            assert native.summary is not None
            ci = next((i for i,row in enumerate(native.summary['collections']) if row['generation']==generation),-1)
            if ci<0:
                raise ContractError('actual committed original collection membership required')
            packet, record = native._original_episode(ci,ordinal,parent=parent,budget_loan=budget_loan,
                episode_id=spec.episode_id,viewer=spec.learner_viewer,aggregate_window=model.event.aggregate_window,
                base_local=base)
        else:
            # Reuse exact captured original evidence from SAME parent. This is
            # a Python backend read; never an alternate reader for Rust.
            # Whole Python original cohort is the current shared reader unit.
            # Bound ALL original read/decode ledger growth before the first read,
            # in addition to protected state/plan copies already admitted.
            control_copy=8*(32*sum(sizes)+65536)
            raw_growth=sum(32*identity[2]+65536 for path,(identity,_) in parent.scan.files.items()
                if path.is_relative_to(plan.output))
            budget_loan.admit_local(base+control_copy+raw_growth)
            owner = (parent.scan.packed_owner(plan.output) if state['schema_version'] == 4 else
                _SourceOwner(plan.output,settings,min(parent.deadline,budget_loan.deadline),
                max_owned_bytes=parent.limit,extra_owned_bytes=parent.resident_bytes+budget_loan.base_bytes+base+control_copy))
            for path,(identity,checksum_value) in parent.scan.files.items():
                if path.is_relative_to(plan.output):
                    owner.remember(path,identity,checksum_value)
            for original_root, patterns in parent.scan._game_source_inventories:
                if original_root==plan.output:
                    owner.inventories.extend(patterns)
            collection = _inspect_component_collection(owner,actions,'train',_game=True)
            _check_collection_parent(collection.manifest,state,plan)
            selected = tuple(row for row in collection.episodes if plan.linkage['learner_seats'][row.episode_id]==row.viewer)
            if len(selected)!=16 or selected[ordinal].episode_id!=spec.episode_id:
                raise ContractError('whole original sixteen selected roles required')
            record = selected[ordinal].records
        if record.end_kind!='natural':
            raise ContractError('natural original selected endpoint required')
        budget_loan.admit_local(base+_owned_bytes(record)+65536)
        borrow = OriginalGameEpisodeBorrow._from_owner(parent,source,budget_loan,episode_id=spec.episode_id,
            record=record,packet=packet,read_owner=owner)
        packet = owner = None
        yield borrow
    except BaseException as error:
        primary=error
        parent._poison(error)
        if borrow is not None:
            borrow._failure = borrow._failure or error
        raise
    finally:
        packet = owner = None
        del state,records,plan,specs
        record=None
        if borrow is not None:
            try:
                borrow.close()
            except BaseException as cleanup:  # actual graph cleanup cannot replace initiating failure
                if primary is None:
                    raise
                vars(primary)['_original_game_episode_cleanup_owner']=borrow
                if cleanup is not primary:
                    try:
                        primary.add_note('original episode retirement: '+str(cleanup))
                    except BaseException:  # noqa: BLE001, S110 - diagnostic failure is best effort
                        pass
            finally:
                if borrow._closed:
                    borrow._finish_scope()


PROJECTION_KEYS = {'recipe', 'parent_run_id', 'batch_id', 'collection_generation', 'expected_optimizer_steps',
                   'expected_policy_version', 'collection_manifest_sha256', 'numeric_manifest_sha256',
                   'numeric_files_sha256', 'source_correspondence'}


def frozen_linkage(state: Json, directory: Path, generation: int, groups: Any, registry: Any) -> Json:
    return {'parent_run_id': state['run_id'], 'parent_run_path': str(directory),
        'batch_id': f'batch-{generation:06}', 'collection_generation': generation,
        'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes},
        'schedule_digest': digest(serialize_schedule(groups, 4)),
        'policy_registry_digest': digest({key: asdict(value) for key, value in registry.items()})}


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
    if binding['model_id'] == 'heuristic_refresh':
        options = parse_refresh_options(binding['policy_options'])
        if (header['model_version'] != '1'
                or header['policy'] != refresh_policy_label(actions, card_instance_rules(), options)
                or canonical(header.get('baseline_options')) != canonical(asdict(options))
                or 'neural' in header or 'tree' in header):
            raise ContractError('original configured baseline identity mismatch')
    elif binding['policy_options'] is None:
        if header['model_version'] != '1' or header['policy'] != binding['model_id'] + '-card-instance-v2':
            raise ContractError('original baseline identity mismatch')
    else:
        raw = manifest['neural_inputs']['policies'][header['policy']]
        safe, label = validate_neural_metadata(raw, actions=actions, rules=card_instance_rules())
        if manifest['schema_version'] == 6:
            # Inference5 is written by atomic_json's canonical UTF-8+newline
            # codec. Retained Journal6 metadata keeps that complete document
            # even after Registry4 deliberately deletes the physical manifest.
            encoded = (canonical(raw['inference_manifest']) + '\n').encode()
            if hashlib.sha256(encoded).hexdigest() != safe['expected_manifest_digest']:
                raise ContractError('HPO original neural manifest bytes/hash differ from generation identity')
        if header['model_version'] != ('4' if manifest['schema_version'] == 6 else '3') or label != header['policy'] or canonical(safe) != canonical(binding['policy_options']) or canonical(header.get('neural')) != canonical(raw):
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
    from gf_common.training_opponents import refresh_opponent, training_opponents
    mixed = training_opponents(state['effective_config'])
    if mixed is not None:
        from gf_training.collection import training_groups_refresh
        actions, _ = _state(state, _game=True)
        name, binding = refresh_opponent(actions, mixed)
        registry[name] = binding
        groups = training_groups_refresh(state['seed'], generation, label, tuple(snapshots), rng,
            actions=actions, training_opponents=mixed)
    else:
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
    formal_control: Any = None
    packed_batch: Any = None
    hpo_job: Any = None

    def _require_hpo_job(self) -> None:
        from gf_training.lineage import require_hpo_run_owner
        require_hpo_run_owner(self.ownership.directory,self.ownership)
        from gf_training.hpo_gpu_lease import require_training_job
        from gf_training.hpo_lineage import hpo_spec
        op=hpo_spec(self.state)['operating']
        require_training_job(self.hpo_job,ledger=self.ledger,run_id=self.state['run_id'],
            global_cuda_bytes=op['managed_cuda_bytes'],max_workers=op['max_participant_workers'])

    def check(self, *, steps: int | None = None) -> None:
        if _hpo(self.state): self._require_hpo_job()
        child, ledger = self.process, self.ledger
        lease = ledger.learner
        if (not self.active or _OWNED_LOCKS.get(self.ownership.directory) != self.ownership.token
                or child.directory.resolve() != self.ownership.directory or lease is None or child.lease is not lease
                or lease._cleanup_owner is not child or child.ledger is not ledger or lease.released or lease.creation_pending
                or child.process is None or child.process.returncode is not None or lease.pid != child.process.pid
                or child.steps != (self.state['optimizer_steps'] if steps is None else steps)
                or canonical(child.hardware) != canonical(self.state['hardware'])):
            raise ContractError('SAME live game parent lock/learner/process/lease/ledger required')

        if self.formal_control is not None:
            self.formal_control.originals.finalize()


def validate_owned_collection(plan: CollectionPlan, ledger: Any, *, max_owned_bytes: int,
                              deadline: float) -> _OnlineGameOwner:
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_gpu_lease import _AuditedLedger
    from gf_training.hpo_learning_process import HpoLearningProcess
    from gf_training.structured_learning_process import GameLearningProcess
    if type(ledger) not in (ReservationLedger,_AuditedLedger) or ledger.learner is None or ledger.workers or ledger.updating:
        raise ContractError('online collection requires idle SAME learner ledger')
    child = ledger.learner._cleanup_owner
    if type(child) not in (GameLearningProcess, HpoLearningProcess) or type(getattr(child, 'online_owner', None)) is not _OnlineGameOwner:
        raise ContractError('online collection requires concrete wire4 live owner')
    owner = child.online_owner
    if type(ledger) is _AuditedLedger and (type(child) is not HpoLearningProcess or not _hpo(owner.state)):
        raise ContractError('audited HPO ledger cannot enter an ordinary Source')
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
    validator = validate_game_parent
    if _hpo(owner.state):
        from gf_training.hpo_source import validate_hpo_parent
        validator = validate_hpo_parent
    state, records = validator(owner.ownership.directory, require_complete=False,
        max_owned_bytes=max_owned_bytes, deadline=deadline)
    if _hpo(state):
        from gf_training.hpo_lineage import hpo_spec, hpo_state
        actions, model, _ = hpo_state(state)
    else:
        actions, model = _state(state, _game=True)
    from gf_training.structured_numeric_worker import wire_identity
    expected_identity = wire_identity(model, InferenceBindingV2(**state['execution']['learner']['binding']),
        digest(asdict(actions)), state['resume_identity'], action_contract=actions.action_contract,
        purpose='qualified_game', synthetic_capacity=False,
        cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'], _actions=actions,
        storage_binding=state.get('storage_binding'), **({'_hpo': True} if _hpo(state) else {}))
    if _hpo(state):
        from gf_training.hpo_config import hpo_training_config
        from gf_training.hpo_numeric_worker import wire_identity as hpo_identity
        expected_identity = hpo_identity(model, hpo_training_config(hpo_spec(state)['training_recipe']),
            InferenceBindingV2(**state['execution']['learner']['binding']), digest(asdict(actions)),
            state['resume_identity'], actions=actions, cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'],
            storage_binding=state.get('storage_binding'),phase_limits={key:hpo_spec(state)['operating'][key]
                for key in ('numeric_budget_bytes','save_budget_bytes')})
    if (canonical(child.identity) != canonical(expected_identity) or canonical(asdict(child.model)) != canonical(asdict(model))
            or canonical(asdict(child.binding)) != canonical(state['execution']['learner']['binding'])
            or child.visibility != state['execution']['learner']['cuda_visible_devices']
            or ledger.learner_bytes != 0 or ledger.learner.resident_bytes != child.resident_bytes):
        raise ContractError('online learner wire/model/execution/reservation differs from original parent')
    planner = recorded_plan
    if _hpo(state):
        from gf_training.hpo_source import recorded_hpo_plan
        planner = recorded_hpo_plan
    expected, _, _ = planner(owner.ownership.directory, state, records, state['updates'], deadline)
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
    _verified_decisions: int = field(init=False, repr=False)
    _verified_collected: int = field(init=False, repr=False)
    _verified_delta: int = field(init=False, repr=False)
    _decoded_retired: bool = field(default=False, init=False, repr=False)
    _decoded_retirement_error: BaseException | None = field(default=None, init=False, repr=False)
    _hpo_save_ack: bytes | None = field(default=None, init=False, repr=False)
    _hpo_ack_process: Any = field(default=None, init=False, repr=False)
    _hpo_ack_capsule: Any = field(default=None, init=False, repr=False)
    _hpo_ack_charge: int = field(default=0, init=False, repr=False)
    _hpo_ack_source_floor: int = field(default=0, init=False, repr=False)
    _hpo_save_phase: str = field(default='empty', init=False, repr=False)
    _hpo_save_error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        hpo = _hpo(self.source.state)
        self._verified_decisions = integer(self.source.selection['used_decisions'] if hpo else self.source.decision_count, 1)
        self._verified_collected = integer(self.source.decision_count, self._verified_decisions)
        self._verified_delta = integer(self.expectation['expected_optimizer_step_delta'] if hpo else
            4 * ((self._verified_decisions + 127) // 128), 1)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('_verified_decisions', '_verified_collected', '_verified_delta') and name in vars(self):
            raise AttributeError('verified issued decision count is immutable')
        if (name in ('_hpo_save_ack', '_hpo_ack_process', '_hpo_ack_capsule', '_hpo_ack_charge', '_hpo_ack_source_floor')
                and vars(self).get('_hpo_save_phase', 'empty') != 'empty'):
            raise AttributeError('acknowledged HPO save binding is immutable')
        if name == '_hpo_save_phase' and vars(self).get(name) == 'failed' and value != 'failed':
            raise AttributeError('failed HPO save admission is terminal')
        object.__setattr__(self, name, value)

    @property
    def expectation(self) -> Json:
        return json.loads(self._expectation)

    def require(self, process: Any, *, after: bool = False) -> None:
        if self._hpo_save_error is not None:
            raise self._hpo_save_error
        if self._decoded_retirement_error is not None:
            raise self._decoded_retirement_error
        if self._decoded_retired and not after:
            raise ContractError('retired decoded projection cannot replay numerical use')
        if (type(self.source) is not _GameBatchSource or type(self.source.online) is not _OnlineGameOwner
                or not self.active or self.source.issued is not self or self.process is not process or self.source.online.issued is not self
                or self.source.online.state['updates'] != self.expectation['collection_generation']):
            raise ContractError('update requires the SAME active Source-issued projection')
        before = self.expectation['expected_optimizer_steps']
        if not self._decoded_retired and (self.source.decision_count != self._verified_collected
                or _hpo(self.source.state) and self.source.selection['used_decisions'] != self._verified_decisions):
            raise ContractError('issued decision count differs from actual decoded Source')
        expected_after = before + self._verified_delta
        self.source.online.check(steps=expected_after if after else before)
        if not after and (self.completed or process.steps != before):
            raise ContractError('replayed/stale Source update transaction')
        if not after:
            self.source.recheck_originals()
            self.source.controls.finalize()
            self.source.online.collection_controls.finalize()

    def _poison_hpo_save(self, error: BaseException) -> None:
        self._hpo_save_phase = 'failed'
        if self._hpo_save_error is None:
            self._hpo_save_error = error
        if self.source.native_inputs is not None:
            self.source.native_inputs._poison_hpo_save(self._hpo_save_error)

    def _record_hpo_save_ack(self, process: Any, reply: Json) -> None:
        """Only the actual completed child request can seal this control value."""
        try:
            if (not _hpo(self.source.state) or self._hpo_save_phase != 'empty'
                    or self.completed or process is not self.process
                    or process._issued_handle is not self or process.ledger.updating):
                raise ContractError('HPO save ACK requires the SAME just-verified child request')
            self.require(process, after=True)
            expected = self.expectation
            from gf_training.hpo_numeric_worker import digest_projection
            if (reply['transaction_id'] != digest_projection(expected, storage_binding=self.source.state.get('storage_binding'))
                    or reply['collection_generation'] != expected['collection_generation']
                    or reply['expected_optimizer_steps'] != expected['expected_optimizer_steps']
                    or reply['artifact']['optimizer_steps'] != process.steps):
                raise ContractError('HPO child save ACK differs from the issued transaction')
            value = {'transaction_id': reply['transaction_id'],
                'expectation_sha256': hashlib.sha256(self._expectation).hexdigest(),
                'collection_generation': reply['collection_generation'],
                'expected_optimizer_steps': reply['expected_optimizer_steps'],
                'artifact': reply['artifact']}
            from gf_models.sharded_arrays import json_bytes
            size = json_bytes(value)
            # Retained canonical bytes and bounded decoded aliases stay Source
            # metadata; never refund them with the consumed token graph.
            charge = 64 * size + 131072
            self.source.controls.charge(charge)
            self._hpo_ack_charge = charge
            self._hpo_ack_source_floor = self.source.controls.retained
            if self.source.snapshot is not None:
                self.source.snapshot.reserve(self.source.snapshot._external_bytes + charge)
            self._hpo_save_ack = canonical(value).encode()
            self._hpo_ack_process = process
            self._hpo_ack_capsule = self.source.native_inputs
            self._hpo_save_phase = 'latched'
        except BaseException as error:
            self._poison_hpo_save(error)
            raise

    def _require_hpo_save_ack(self, process: Any, generation: int, files: Json) -> Json:
        self.require(process, after=True)
        control = self.source.controls
        if (not _hpo(self.source.state) or self._hpo_save_ack is None
                or self._hpo_ack_process is not process or self._hpo_ack_capsule is not self.source.native_inputs
                or not self.completed or self.prepared or process._issued_handle is not None
                or process.ledger.updating or self._hpo_save_phase not in ('latched', 'admitting')
                or control._closed or control.retained < self._hpo_ack_source_floor
                or control._packed_pool is not None and control._pool_control_charge < control.retained):
            raise ContractError('SAME completed HPO child ACK/projection/capsule required')
        ack = json.loads(self._hpo_save_ack)
        expected = self.expectation
        from gf_training.hpo_numeric_worker import digest_projection
        if (generation != expected['collection_generation'] + 1
                or ack['collection_generation'] != expected['collection_generation']
                or ack['transaction_id'] != digest_projection(expected, storage_binding=self.source.state.get('storage_binding'))
                or ack['expectation_sha256'] != hashlib.sha256(self._expectation).hexdigest()
                or ack['expected_optimizer_steps'] != expected['expected_optimizer_steps']
                or ack['artifact']['optimizer_steps'] != expected['expected_optimizer_steps'] + self._verified_delta
                or process.steps != ack['artifact']['optimizer_steps']
                or canonical(files) != canonical(ack['artifact']['files'])
                or files['inference.json'] != ack['artifact']['inference_digest']
                or files['resume.json'] != ack['artifact']['resume_digest']):
            raise ContractError('HPO saved pair/map/generation differs from the immutable child ACK')
        return ack

    def _admit_hpo_saved_output(self, process: Any, generation: int, files: Json, ownership: Any) -> None:
        try:
            if self._hpo_save_error is not None:
                raise self._hpo_save_error
            from gf_training.lineage import require_hpo_run_owner
            if self._hpo_save_phase != 'latched' or ownership is not self.source.online.ownership:
                raise ContractError('one SAME-owner HPO acknowledged save admission required')
            require_hpo_run_owner(self.source.controls.root, ownership)
            self._require_hpo_save_ack(process, generation, files)
            self._hpo_save_phase = 'admitting'
            if self.source.native_inputs is not None:
                self.source.native_inputs.caller_growth(self.source.controls.retained - self.source._control_base)
                self.source.native_inputs._admit_hpo_saved_output(self, generation, files)
            self._hpo_save_phase = 'admitted'
        except BaseException as error:
            self._poison_hpo_save(error)
            raise

    def retire_packed_decoded(self) -> None:
        """One guarded consumed-graph boundary; physical originals remain live."""
        from gf_training.group_pack_game_source import PackedBatchCustody, PackedSourceOwner
        source = self.source
        if type(source) is not _GameBatchSource:
            raise ContractError('SAME issued game Source required for decoded retirement')
        if not _packed(source.state) or source.native_inputs is not None:
            return
        if self._decoded_retirement_error is not None:
            raise self._decoded_retirement_error
        if not self.completed or self.prepared or self._decoded_retired:
            raise ContractError('decoded retirement requires one completed unprepared update')
        if (type(source.online) is not _OnlineGameOwner or self.process is not source.online.process
                or type(source.snapshot) is not _SourceSnapshot):
            raise ContractError('SAME actual packed Python Source/process required')
        custody = source.online.packed_batch
        if (type(custody) is not PackedBatchCustody or custody.retired
                or type(custody.owner) is not PackedSourceOwner
                or source.snapshot._owner is not custody.owner):
            raise ContractError('SAME retained packed original owner required')
        owner = custody.owner
        self.require(self.process, after=True)
        if (self.process._issued_handle is not None or source.online.ledger.updating
                or source.online.ledger.workers or any(s._loans for s in owner.pool.sources)):
            raise ContractError('learner request, numeric update or group borrow remains active')
        source.recheck_originals()
        try:
            # Do not retain a local alias of either consumed graph across refund.
            source.records = None
            source.snapshot.collection = replace(source.snapshot.collection, episodes=())
            source.snapshot._retained_bytes = _owned_bytes((source.snapshot.collection._manifest,
                source.snapshot.collection._proof, owner.files, owner.inventories))
            source.snapshot._external_bytes = 0
            numeric_metadata = None if source._numeric is None else (str(source._numeric[0]), *source._numeric[1:])
            # These serialized values can still be decoded by later reportable
            # guards. Preserve their existing conservative JSON allowances too.
            parsing = 32*(len(source.snapshot.collection._manifest)+len(source.snapshot.collection._proof))+131072
            remaining = source.snapshot._retained_bytes + parsing + self._hpo_ack_charge + _owned_bytes((self._expectation, self.files, numeric_metadata, source._hpo_selection, self._hpo_save_ack))
            owner.retire_decoded(retained_bytes=remaining)
        except BaseException as error:
            self._decoded_retirement_error = error
            vars(error)['_source_projection_cleanup_owner'] = self
            raise
        self._decoded_retired = True

    def prepare_commit(self) -> None:
        try:
            self._prepare_commit()
        except BaseException as error:
            if _hpo(self.source.state):
                self._poison_hpo_save(error)
            raise

    def _prepare_commit(self) -> None:
        self.require(self.process, after=True)
        if _hpo(self.source.state) and self._hpo_save_phase != 'admitted':
            raise ContractError('HPO preparation requires the actual acknowledged saved pair')
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
        if self.source.online.packed_batch is not None:
            proof = self.source.proof
            # Optimizer is complete: explicitly retire the bounded decoded input
            # graph before original pool/FD/budget retirement.
            if self.source.native_inputs is None:
                if not self._decoded_retired:
                    self.retire_packed_decoded()
            else:
                # Native owns its separate existing payload-retirement boundary.
                # ACK metadata belongs to source.controls and its independent
                # shared control reservation, not this consumed native graph.
                self.source.online.packed_batch.owner.retire_decoded()
            self.source.online.packed_batch.prepare(proof)
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
    _hpo_selection: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from gf_training.native_card_instance import _NativeGameInputs
        if self.native_inputs is None:
            if type(self.snapshot) is not _SourceSnapshot or type(self.records) is not tuple:
                raise ContractError('Python game Source requires its concrete snapshot/records')
        elif (type(self.native_inputs) is not _NativeGameInputs or self.snapshot is not None or self.records is not None):
            raise ContractError('native game inputs are a mutually exclusive concrete representation')
        self._control_base = self.controls.retained
        if _hpo(self.state):
            from gf_training.hpo_collection import selection_for_counts
            counts = (tuple(self.native_inputs.collection['selection']['episode_counts']) if self.native_inputs is not None
                else tuple(len(record.frames) for record in cast(tuple[Any, ...], self.records)))
            context = self.plan.linkage['selection_context']
            if self.snapshot is not None:
                # Worst-case list/set/JSON/RNG scratch is admitted before sampling.
                self.snapshot.reserve(self.snapshot._external_bytes + 256 * sum(counts) + 65536)
            selection = selection_for_counts(counts, **{key: context[key]
                for key in ('learning_seed', 'batch', 'decision_budget', 'used_before')})
            self._hpo_selection = canonical(selection).encode()

    @property
    def selection(self) -> Json:
        if self._hpo_selection is None:
            raise ContractError('only explicit HPO Source has selected endpoint facts')
        return json.loads(self._hpo_selection)

    @property
    def _selection_owned(self) -> int:
        return 0 if self._hpo_selection is None else 32 * len(self._hpo_selection) + 65536

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
        proof = self.snapshot.collection.proof
        if _hpo(self.state):
            from gf_common.hpo import PROOF_KINDS, optimizer_delta
            from gf_training.hpo_lineage import hpo_spec
            recipe = hpo_spec(self.state)['training_recipe']
            proof.update(schema_version=3, kind=PROOF_KINDS[int(_packed(self.state))], selection=self.selection,
                expected_optimizer_step_delta=optimizer_delta(self.selection['used_decisions'],
                    epochs=recipe['epochs'], minibatch_size=recipe['minibatch_size']))
        return proof

    def correspondence(self) -> Json:
        from gf_training.lineage import generation_path
        generation = self.plan.linkage['collection_generation']
        immutable = {str(path.relative_to(self.controls.root)): self.controls.verified_digest(path) for path in (
            generation_path(self.controls.root, generation),)}
        inference_path = checkpoint_path(self.controls.root, generation) / 'inference.json'
        if _hpo(self.state):
            from gf_training.hpo_source import original_inference
            record = self.controls.object(generation_path(self.controls.root, generation))
            _, inference_sha = original_inference(self.controls, self.state, record, generation, live=self.online is not None)
        else:
            inference_sha = self.controls.verified_digest(inference_path)
        immutable[str(inference_path.relative_to(self.controls.root))] = inference_sha
        content = {'run_id': self.state['run_id'], 'config_sha256': self.state['config_sha256'],
            'identity': self.state['identity'], 'execution': self.state['execution'], 'plan': self.plan.linkage, 'files': immutable,
            'source_set_sha256': self.proof['source_set_sha256']}
        if _packed(self.state):
            evidence = getattr(self, '_historical_proof', self.proof)
            content.update(recipe='card-instance-packed-source-controls-v1', storage=evidence['storage'], custody=evidence['custody'])
        if _hpo(self.state):
            content.update(recipe='card-instance-hpo-source-controls-v1', selection=self.selection,
                hpo_spec_sha256=self.state['hpo_spec_sha256'])
        fingerprint = digest(content)
        expected = _source_correspondence(fingerprint, self.declaration, self.episode_digests, game=True, storage_binding=self.state.get('storage_binding'), hpo=_hpo(self.state))
        if self.native_inputs is not None and canonical(expected) != canonical(self.native_inputs.correspondence):
            raise ContractError('native correspondence differs from SAME Source control fingerprint')
        return expected

    def write_numeric(self, directory: Path) -> Json:
        if self.native_inputs is not None:
            self.native_inputs.caller_growth(self.controls.retained - self._control_base)
            return self.native_inputs.publish_numeric(directory)
        assert self.snapshot is not None and self.records is not None
        extra = self.snapshot.extra_bytes_for(self.records) + self._selection_owned
        predicted=_predicted_token_bytes(self.records)
        numeric_limit=None
        if _hpo(self.state):
            from gf_training.hpo_lineage import hpo_spec
            numeric_limit=hpo_spec(self.state)['operating']['numeric_budget_bytes']
            if predicted>numeric_limit:
                raise ContractError('HPO numeric token envelope exceeded before allocation')
        self.snapshot.reserve(self.snapshot.additional_bytes((self.records, self.model))
            + predicted + self._selection_owned)
        rollout: Any
        extra_context: Json
        if _hpo(self.state):
            from gf_training.hpo_collection import replay_selection
            from gf_training.hpo_config import hpo_training_config
            from gf_training.hpo_lineage import hpo_spec
            from gf_training.hpo_rollout import HpoRollout
            recipe = hpo_training_config(hpo_spec(self.state)['training_recipe'])
            rollout = HpoRollout(self.records, self.model, recipe, actions=self.actions, rules=card_instance_rules(),
                selected_locations=replay_selection(self.selection, tuple(len(record.frames) for record in self.records)), _reserved_bytes=extra)
            from gf_training.hpo_rollout import _numeric_context
            extra_context = {'_hpo': _numeric_context(rollout, self.selection)}
        else:
            rollout = game_rollout(self.records, self.model, actions=self.actions, rules=card_instance_rules(), reserved_bytes=extra)
            extra_context = {}
        additional=self.snapshot.additional_bytes((rollout.model,rollout.episodes,rollout.locations))
        snapshot=self.snapshot
        def numeric_reserve(peak: int) -> None:
            if numeric_limit is not None and additional+max(0,peak-rollout.resident_bytes)>numeric_limit:
                raise ContractError('HPO additional numeric encoding envelope exceeded')
            snapshot.reserve(peak)
        return _write_rollout(directory, rollout, **extra_context, max_owned_bytes=self.snapshot.max_owned_bytes,
            extra_owned_bytes=self.snapshot.extra_bytes_for((rollout.model, rollout.episodes, rollout.locations,
                rollout.action_config_hash, rollout.policy_version, rollout.input_binding)) + self._selection_owned,
            check=self.snapshot.check, reserve=numeric_reserve)

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
        if _hpo(self.state):
            from gf_training.hpo_lineage import hpo_spec
            numeric_limit = hpo_spec(self.state)['operating']['numeric_budget_bytes']
            # Seal the SAME file/hash/stat snapshot before admitting its JSON.
            # The retained object remains charged to the existing Source scan.
            with self.controls.file(directory / 'rollout.json', expected_digest) as (_, info):
                envelope = 32 * info.st_size + 65536
                if envelope > numeric_limit:
                    raise ContractError('HPO numeric JSON envelope exceeded before parse')
                self.snapshot.reserve(prior + envelope)
                manifest = self.controls.object(directory / 'rollout.json', expected_digest)
        else:
            self.snapshot.reserve(prior + 32 * (directory / 'rollout.json').lstat().st_size + 65536)
            manifest = self.controls.object(directory / 'rollout.json', expected_digest)
        growth = self.controls.retained - self._control_base
        self.snapshot.reserve(prior + growth)
        reader: Any = read_game_rollout
        parameters: Json = {'declaration': self.declaration}
        if _hpo(self.state):
            from gf_training.hpo_config import hpo_training_config
            from gf_training.hpo_lineage import hpo_spec
            from gf_training.hpo_rollout import read_hpo_rollout
            reader = read_hpo_rollout
            parameters = {'training': hpo_training_config(hpo_spec(self.state)['training_recipe']), 'expected_selection': self.selection,
                'numeric_budget_bytes':hpo_spec(self.state)['operating']['numeric_budget_bytes']}
        rollout = reader(directory, expected_digest, self.model, actions=self.actions, rules=card_instance_rules(),
            **parameters, expected_correspondence=self.correspondence(),
            max_owned_bytes=self.snapshot.max_owned_bytes, extra_owned_bytes=self.snapshot.resident_bytes + growth + self._selection_owned,
            check=self.snapshot.check, reserve=lambda peak: snapshot.reserve(peak + growth + self._selection_owned),
            storage_binding=self.state.get('storage_binding'))
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
        if _hpo(self.state):
            from gf_training.hpo_source import require_hpo_production
            require_hpo_production(self.state)
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
        expected = {'recipe': 'card-instance-packed-source-update-v1' if self.state['schema_version'] == 4 else 'card-instance-source-update-v1', 'parent_run_id': self.state['run_id'],
            'batch_id': self.plan.linkage['batch_id'], 'collection_generation': generation,
            'expected_optimizer_steps': self.state['optimizer_steps'], 'expected_policy_version': self.learner_label,
            'collection_manifest_sha256': self.proof['manifest_sha256'], 'numeric_manifest_sha256': files['rollout.json'],
            'numeric_files_sha256': digest(dict(sorted(files.items()))), 'source_correspondence': self.correspondence()}
        if _hpo(self.state):
            expected.update(recipe='card-instance-hpo-source-update-v1', training_recipe=self.state['effective_config']['hpo_spec']['training_recipe'],
                selection=self.selection, collected_decisions=self.decision_count, used_decisions=self.selection['used_decisions'],
                expected_optimizer_step_delta=self.proof['expected_optimizer_step_delta'])
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
        hpo = _hpo(state)
        if hpo:
            from gf_training.hpo_lineage import hpo_spec, hpo_state
            from gf_training.hpo_source import recorded_hpo_plan
            actions, model, _ = hpo_state(state)
            planner = recorded_hpo_plan
            semantic_keys = SEMANTIC_KEYS_V3 - {'reward_policy_version'}
        else:
            actions, model = _state(state, _game=True)
            planner = recorded_plan
            semantic_keys = SEMANTIC_KEYS_V3
        generation = plan.linkage['collection_generation']
        expected, label, _ = planner(controls.root, state, records, generation, deadline)
        if canonical(plan.linkage) != canonical(expected.linkage):
            raise ContractError('original batch plan differs from committed generation')
        if hpo:
            from gf_training.hpo_source import original_inference
            previous, _ = original_inference(controls, state, records[generation], generation, live=online is not None)
        else:
            previous = controls.object(checkpoint_path(controls.root, generation) / 'inference.json', records[generation]['inference_sha256'])
        native_inputs = None
        from gf_common.training_opponents import training_opponents
        if (hpo or settings.backend == 'rust') and training_opponents(state['effective_config']) is not None:
            from gf_training.native_backend import require_native_refresh_opponents
            require_native_refresh_opponents()
        if settings.backend == 'rust':
            if any(binding.model_id == 'heuristic_refresh' for binding in plan.bindings.values()):
                from gf_training.native_backend import require_native_refresh_baseline
                require_native_refresh_baseline()
            if hpo:
                from gf_training.native_backend import hpo_reader_settings
                hpo_reader_settings(settings, packed=_packed(state))
            from gf_training.native_card_instance import _NativeGameInputs
            provider = None
            if _packed(state):
                from gf_training.native_backend import group_pack_reader_settings
                from gf_training.native_group_pack import OriginalArtifactProvider
                group_pack_reader_settings(settings)
                # Register actual current custody first so historical owners join
                # its SAME pool; no second original Source/pool is constructed.
                if online is not None:
                    controls.packed_owner(plan.output, custody=online.packed_batch)
                for gi in range(state['updates']):
                    controls.packed_owner(controls.root / 'collections' / f'batch-{gi:06}')
                provider = scope.enter_context(OriginalArtifactProvider(tuple(controls._packed_owners.values())))
            mode = 'live_cohort' if online is not None else 'historical_parent'
            native_plan = {'schema_version': 3, 'mode': mode,
                'parents': [{'directory': str(controls.root), 'require_complete': False}],
                'collections': ([{'directory': str(plan.output), 'parent_index': 0}] if online is not None else [])}
            if online is not None:
                native_plan['cohort_collection_index'] = 0
            if provider is not None:
                native_plan.update(schema_version=4, storage_binding=state['storage_binding'],
                    artifact_registry_sha256=provider.registry_sha256)
            if hpo:
                native_plan['schema_version'] = 5
            from gf_training.ppo_run import owned_bytes
            external = ()
            if provider is not None:
                root = Path(state['effective_config']['storage']['db_root'])
                external = (root / state['run_id'], *(root / name for name in ('catalog.sqlite3',
                    'catalog.sqlite3-wal', 'catalog.sqlite3-shm', 'catalog.sqlite3-journal')))
            remaining_storage = max(0, state['effective_config']['training']['max_owned_bytes']
                - owned_bytes(controls.root, deadline, external_roots=external))
            native_inputs = scope.enter_context(_NativeGameInputs(native_plan, settings, actions=actions,
                max_owned_bytes=max_owned_bytes, deadline=deadline, extra_owned_bytes=controls.retained + extra_owned_bytes,
                max_scratch_bytes=remaining_storage, artifact_provider=provider, shared_controls=controls if provider is not None else None))
            assert native_inputs.summary is not None
            native_inputs.collection_index = next((index for index, row in enumerate(native_inputs.summary['collections'])
                if row['generation'] == generation), -1)
            if (canonical(native_inputs.collection['linkage']) != canonical(plan.linkage)
                    or native_inputs.collection['learner_label'] != label
                    or canonical(asdict(native_inputs.models[0])) != canonical(asdict(model))
                    or canonical(native_inputs.collection['semantics']) != canonical({key: previous[key] for key in semantic_keys})):
                raise ContractError('native originals differ from SAME recorded Source plan/model/declaration')
            source = _GameBatchSource(None, controls, state, plan, model, actions,
                {key: previous[key] for key in semantic_keys}, label, online, None, native_inputs=native_inputs)
            source.correspondence()
        else:
            if _packed(state):
                original = controls.packed_owner(plan.output, custody=online.packed_batch if online is not None else None)
                original.charge(controls.retained + extra_owned_bytes)
            else:
                original = _SourceOwner(plan.output, settings, deadline, extra_owned_bytes=controls.retained + extra_owned_bytes,
                                        max_owned_bytes=max_owned_bytes)
            role_options = {}
            if hpo:
                from gf_common.card_instance_collection import hpo_batch_role
                role_options['_game_role'] = hpo_batch_role(hpo_spec(state)['training_recipe']['collection_games'])
            collection = _inspect_component_collection(original, actions, 'train', _game=True, **role_options)
            snapshot = _SourceSnapshot(collection, original, _owned_bytes((collection.episodes, collection._manifest, collection._proof, original.files, original.inventories)))
            _check_collection_parent(collection.manifest, state, plan)
            seats = plan.linkage['learner_seats']
            selected = tuple(row for row in collection.episodes if seats[row.episode_id] == row.viewer)
            if len(selected) != (hpo_spec(state)['training_recipe']['collection_games'] if hpo else 16) or any(row.model_id != model.model_id or row.model_version != ('4' if hpo else '3') for row in selected):
                raise ContractError('all16 original neural learner seats required')
            source = _GameBatchSource(snapshot, controls, state, plan, model, actions,
                {key: previous[key] for key in semantic_keys}, label, online, tuple(row.records for row in selected))
            del collection, selected
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
            if _packed(state) and online is None and source.snapshot is not None:
                # No historical decoded cohort remains resident after this scope.
                assert source.snapshot is not None
                original = source.snapshot._owner
                source.records = None
                source.snapshot = None
                from gf_training.group_pack_game_source import PackedSourceOwner
                cast(PackedSourceOwner, original).retire_decoded()
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
    with controls:
        controls.configure_packed(online.state)
        if online.packed_batch is not None:
            online.packed_batch.borrow(controls)
        if _hpo(online.state):
            from gf_training.hpo_lineage import validate_hpo_chain
            state, records = validate_hpo_chain(online.ownership.directory, require_complete=False,
                max_owned_bytes=max_owned_bytes, deadline=deadline, scan=controls)
        else:
            state, records = _validate_component_chain(online.ownership.directory, require_complete=False,
                require_retained=False, scan=controls, _game=True)
        settings = preparation_config(state['effective_config']['preprocessing'])
        with _open_batch(plan, state, records, controls, settings, online=online,
            max_owned_bytes=max_owned_bytes, deadline=deadline, extra_owned_bytes=extra_owned_bytes) as source:
            yield source


@contextmanager
def _open_game_parent_inputs(directory: Path, *, require_complete: bool = True, max_owned_bytes: int,
                         deadline: float, extra_owned_bytes: int = 0, _scan: _ChainScan | None = None) -> Iterator[tuple[Json, list[Json], Any]]:
    directory = directory.resolve()
    scan = _scan or _ChainScan(directory=directory, max_owned_bytes=max_owned_bytes - extra_owned_bytes, deadline=deadline)
    if type(scan) is not _ChainScan or scan.root != directory or scan.limit > max_owned_bytes - extra_owned_bytes:
        raise ContractError("same bounded original game-parent snapshot required")
    state, records = _validate_component_chain(directory, require_complete=require_complete,
        require_retained=False, scan=scan, _game=True)
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(state['seed']))
    if canonical(records[0]['parent_controls']) != canonical({'schedule_opponent_rng': rng.bit_generator.state,
            'snapshot_pool': [0], 'collection_position': 0}):
        raise ContractError('original generation0 PCG64 differs from frozen seed')
    settings = preparation_config(state['effective_config']['preprocessing'])
    if settings.backend == 'rust':
        from gf_common.training_opponents import training_opponents
        if training_opponents(state['effective_config']) is not None:
            from gf_training.native_backend import require_native_refresh_opponents
            require_native_refresh_opponents()
        if 'run_variant' in state:
            from gf_training.card_instance_formal import require_native_formal_controls
            from gf_training.native_backend import load_native
            require_native_formal_controls(load_native(), packed=state['schema_version'] == 4)
        from gf_training.native_card_instance import _NativeGameInputs
        native_plan = {'schema_version': 3, 'mode': 'historical_parent',
            'parents': [{'directory': str(directory), 'require_complete': require_complete}], 'collections': []}
        with ExitStack() as native_scope:
            provider = None
            if state['schema_version'] == 4:
                from gf_training.native_backend import group_pack_reader_settings
                from gf_training.native_group_pack import OriginalArtifactProvider
                group_pack_reader_settings(settings)
                for generation in range(state['updates']):
                    scan.packed_owner(directory / 'collections' / f'batch-{generation:06}')
                if state['updates'] == 0:
                    if (scan._packed_state is not state or len(records) != 1
                            or type(records[0].get('generation')) is not int or records[0]['generation'] != 0
                            or records[0].get('update_sha256') is not None
                            or scan._packed_owners or scan._packed_borrows):
                        raise ContractError('actual uncommitted generation0 parent required before empty native allocation')
                    provider = native_scope.enter_context(OriginalArtifactProvider((),
                        pool=scan.ensure_packed_pool(), shared_controls=scan))
                else:
                    provider = native_scope.enter_context(OriginalArtifactProvider(tuple(scan._packed_owners.values())))
                native_plan.update(schema_version=4, storage_binding=state['storage_binding'],
                    artifact_registry_sha256=provider.registry_sha256)
            inputs = native_scope.enter_context(_NativeGameInputs(native_plan, settings,
                actions=_state(state, _game=True)[0], max_owned_bytes=max_owned_bytes, deadline=deadline,
                extra_owned_bytes=scan.retained + extra_owned_bytes, artifact_provider=provider,
                shared_controls=scan if provider is not None else None))
            assert inputs.summary is not None
            parent = inputs.summary['parent']
            if (parent['identity_sha256'] != digest(state['identity'])
                    or parent['run_sha256'] != scan.files[directory / 'run.json'][1]
                    or any(canonical(state[key]) != canonical(value) for key, value in parent['state'].items())):
                raise ContractError('native historical parent differs from original game control state')
            yield state, records, inputs
            scan.finalize()
        return
    source_inventories = []
    for number in range(1, len(records)):
        plan, _, after = recorded_plan(directory, state, records, number - 1, deadline)
        if canonical(after) != canonical(records[number]['parent_controls']['schedule_opponent_rng']):
            raise ContractError('original schedule RNG advance mismatch')
        update = scan.object(directory / 'updates' / f'{number:06}.json', records[number]['update_sha256'])
        with _open_batch(plan, state, records, scan, settings, online=None, max_owned_bytes=max_owned_bytes,
                         deadline=deadline, extra_owned_bytes=extra_owned_bytes) as source:
            from gf_common.group_pack_game import same_packed_proof
            if not (same_packed_proof(source.proof, update['source_proof']) if state['schema_version'] == 4
                    else canonical(source.proof) == canonical(update['source_proof'])):
                raise ContractError('original Source proof differs from committed generation')
            if state['schema_version'] == 4:
                vars(source)['_historical_proof'] = update['source_proof']
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
                if state['schema_version'] == 4:
                    if path in scan.logical_files and scan.logical_files[path] != entry:
                        raise ContractError('original logical Source changed across snapshots')
                    scan.logical_files[path] = entry
                else:
                    snapshot = (tuple(entry['identity']), entry['sha256'])
                    if path in scan.files and scan.files[path] != snapshot:
                        raise ContractError('original Source changed across parent/batch snapshots')
                    scan.files[path] = snapshot
            scan.charge(_owned_bytes(original.inventories) + 512 + 4 * len(str(original.root)))
            if state['schema_version'] != 4:
                source_inventories.append((original.root, tuple(original.inventories)))
    scan._game_source_inventories = source_inventories
    scan._game_source_settings = settings
    scan._game_source_extra_bytes = extra_owned_bytes
    scan._game_source_max_owned_bytes = max_owned_bytes
    try:
        yield state, records, None
    finally:
        recheck_game_parent_snapshot(scan)


def validate_game_parent(directory: Path, *, require_complete: bool = True, max_owned_bytes: int,
                         deadline: float, extra_owned_bytes: int = 0, _scan: _ChainScan | None = None) -> tuple[Json, list[Json]]:
    """Existing one-shot validator; the private context also supports retained owners."""
    scan = _scan or _ChainScan(directory=directory.resolve(), max_owned_bytes=max_owned_bytes - extra_owned_bytes, deadline=deadline)
    with ExitStack() as scope:
        if _scan is None:
            scope.enter_context(scan)
        with _open_game_parent_inputs(directory, require_complete=require_complete, max_owned_bytes=max_owned_bytes,
                deadline=deadline, extra_owned_bytes=extra_owned_bytes, _scan=scan) as (state, records, _inputs):
            return state, records


def recheck_game_parent_snapshot(scan: Any) -> None:
    """Retain exactly the original Source inventories for an outer read-only owner."""
    def inventories() -> None:
        for original_root, patterns in getattr(scan, '_game_source_inventories', ()):
            comparison = _SourceOwner(original_root, scan._game_source_settings, scan.deadline,
                extra_owned_bytes=scan.retained + scan._game_source_extra_bytes,
                max_owned_bytes=scan._game_source_max_owned_bytes)
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
