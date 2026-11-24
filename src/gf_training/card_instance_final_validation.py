"""Actor-only final64 collection and retained originals; no Dataset/update authority."""
from __future__ import annotations

import json
import math
import stat
import threading
import time
import traceback as traceback_module
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Self

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_collection import (
    FINAL_VALIDATION,
    checked_invocation,
    checked_linkage,
    checked_projection,
    config_projection,
    validation_invocation,
)
from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    closed,
    digest,
    integer,
)
from gf_common.schedule import serialize_schedule
from gf_models.config import PPOTrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_collection import CollectionPublication
from gf_training.card_instance_formal import (
    FinalSourceBudgetLoan,
    FormalGameParent,
    formal_preflight,
)
from gf_training.collection import validation_groups
from gf_training.structured_rollout import EpisodeRecords, _owned_bytes
from gf_training.structured_source import (
    _inspect_component_collection,
    _SourceOwner,
    _SourceSnapshot,
)
from gf_training.structured_tree_validation import _checked_returned_report


def _note(error: BaseException, label: str, secondary: BaseException) -> None:
    try:
        error.add_note(label + str(secondary))
    except BaseException:  # noqa: BLE001, S110 - diagnostics never replace the initiating failure
        pass


# Resource exclusion only: values never retain parent/context; actual weak keys
# prevent ID/path/hash reuse from transferring a borrow to another parent.
_BORROWS: weakref.WeakKeyDictionary[FormalGameParent, object] = weakref.WeakKeyDictionary()
_BORROW_LOCK = threading.Lock()


def _acquire_borrow(parent: FormalGameParent) -> object:
    if type(parent) is not FormalGameParent:
        raise ContractError('SAME concrete active original FormalGameParent required')
    parent.require_active()
    with _BORROW_LOCK:
        if parent in _BORROWS:
            raise ContractError('SAME parent already has an actor64 resource borrow')
        token = object()
        _BORROWS[parent] = token
        return token


def _release_borrow(parent: FormalGameParent, token: object) -> None:
    with _BORROW_LOCK:
        if _BORROWS.get(parent) is not token:
            raise ContractError('SAME actor64 resource borrow token required for retirement')
        del _BORROWS[parent]


def _require_borrow(parent: FormalGameParent, token: object | None = None) -> object:
    """Actual resource exclusion only; no session/Source permission is returned."""
    if type(parent) is not FormalGameParent:
        raise ContractError('SAME concrete retained formal parent required')
    with _BORROW_LOCK:
        current = _BORROWS.get(parent)
        if current is None or token is not None and current is not token:
            raise ContractError('SAME actual actor64 resource borrow required')
        return current


@dataclass(frozen=True)
class GameFinalValidationPlan:
    groups: tuple[Any, ...]
    bindings: dict[str, RuntimePolicyBinding]
    output: Path
    linkage: Json
    config_projection: Json
    validation_invocation: Json
    admission_deadline: float
    execution_deadline: float
    wall_deadline: float
    started_monotonic: float
    kind: str = 'ppo_final_validation'


@dataclass(frozen=True)
class _Final64OriginalContract:
    parent: FormalGameParent
    effective_bytes: bytes
    input_origin_bytes: bytes
    projected_effective_bytes: bytes
    actions: ActionConfigV3
    settings: Any
    expected_facts_bytes: bytes
    recorded_build: Json | None = None


def _admit_source(parent: FormalGameParent, loan: FinalSourceBudgetLoan | None, amount: int) -> None:
    if loan is None:
        parent.admit_external(amount)
    else:
        loan.require(parent)
        loan.admit_local(amount)


def _original_contract(parent: FormalGameParent, *, state: Json | None = None,
                       budget_loan: FinalSourceBudgetLoan | None = None) -> _Final64OriginalContract:
    from gf_common.card_instance import action_config_v3
    from gf_common.preprocessing import PreparationConfig, preparation_config
    from gf_training.native_backend import recorded_reader_build
    parent.require_active()
    sizes = parent.canonical_control_sizes
    if budget_loan is not None:
        budget_loan.require(parent)
        budget_loan.admit_local(8*(32*sum(sizes)+65536))
    parent.recheck()
    original = parent.state if state is None else state
    if original is None or digest(original['effective_config']) != original['config_sha256']:
        raise ContractError('protected original effective Config digest required')
    label, binding = parent.final_binding()
    if type(binding) is not RuntimePolicyBinding or binding.policy_options is None:
        raise ContractError('actual retained final policy binding required')
    effective = original['effective_config']
    actions = action_config_v3(effective['actions'])
    raw_settings = closed(effective['preprocessing'], {field.name for field in fields(PreparationConfig)})
    settings = preparation_config(raw_settings)
    if canonical(asdict(settings)) != canonical(raw_settings):
        raise ContractError('original preprocessing must roundtrip without defaults')
    if effective['run']['split'] != 'train':
        raise ContractError('original formal train Config required')
    projected = {**effective, 'run':{**effective['run'], 'split':'validation'}}
    facts = {'parent_run_id':original['run_id'], 'parent_run_path':str(parent.directory),
        'seed':original['seed'], 'final_generation':original['updates'],
        'final_binding':{'label':label, 'binding':asdict(binding)},
        'code':original['identity']['code'], 'integration':original['identity']['integration']}
    contract = _Final64OriginalContract(parent, canonical(effective).encode(),
        canonical(original['config_inputs']).encode(), canonical(projected).encode(), actions,
        settings, canonical(facts).encode(), recorded_reader_build(original) if settings.backend == 'rust' else None)
    if budget_loan is not None:
        budget_loan._keep_opening(contract)
    return contract


def _plan_original_contract(contract: _Final64OriginalContract, directory: Path, invocation: Json) -> GameFinalValidationPlan:
    contract.parent.recheck()
    facts = json.loads(contract.expected_facts_bytes)
    label = facts['final_binding']['label']
    binding = RuntimePolicyBinding(**facts['final_binding']['binding'])
    assert binding.policy_options is not None
    groups = validation_groups(facts['seed'], label, actions_version=3)
    registry = _registry(label, binding)
    linkage = {'parent_run_id':facts['parent_run_id'], 'parent_run_path':facts['parent_run_path'],
        'batch_id':'validation-final', 'final_generation':facts['final_generation'],
        'final_inference_sha256':binding.policy_options['expected_manifest_digest'], 'policy_label':label,
        'learner_seats':{spec.episode_id:spec.learner_viewer for group in groups for spec in group.episodes},
        'schedule_digest':digest(serialize_schedule(groups, 4)),
        'policy_registry_digest':digest({key:asdict(value) for key,value in registry.items()})}
    return GameFinalValidationPlan(groups, registry, directory, linkage,
        config_projection(json.loads(contract.effective_bytes), json.loads(contract.input_origin_bytes),
            json.loads(contract.projected_effective_bytes)), invocation,
        invocation['admission_deadline_monotonic'], invocation['execution_deadline_monotonic'],
        invocation['wall_deadline_monotonic'], invocation['started_monotonic'])


def require_actor_backend(loaded: Any, *, recorded_build: Json | None = None) -> None:
    """No Rust64 activity before its separate reviewed retained-source adoption."""
    config = loaded.config
    if config.preprocessing.reader != 'participant-structured-v2':
        raise ContractError('explicit participant-structured-v2 final validation reader required')
    if config.preprocessing.backend == 'python':
        return
    if config.preprocessing.backend != 'rust':
        raise ContractError('explicit final validation Python/Rust backend required')
    from gf_training.native_card_instance import _require_final_validation_native
    _require_final_validation_native(config.preprocessing, packed=hasattr(config, 'storage'),
        recorded_build=recorded_build)


def _caller_contract(loaded: Any) -> None:
    cfg = loaded.config
    if (type(cfg.actions) is not ActionConfigV3 or type(cfg.model) is not ConfiguredNeuralModelV3
            or not isinstance(cfg.training, PPOTrainingConfig) or cfg.model.checkpoint is not None
            or cfg.run.split != 'train' or cfg.server.external_url is not None):
        raise ContractError('original corrected formal NN3 training Config required')


def _loaded(loaded: Any, *, recorded_build: Json | None = None) -> None:
    _caller_contract(loaded)
    formal_preflight(loaded)
    require_actor_backend(loaded, recorded_build=recorded_build)

_CONTROL_SCRATCH = 65536


def _wire_upper(value: Any, active: set[int] | None = None) -> int:
    """Size-only bound over existing values; never serialize or copy containers.

    Count repeated references per occurrence because asdict/JSON materialize
    them repeatedly. Six bytes per code point covers UTF8 and JSON escapes.
    """
    if value is None:
        return 4
    if type(value) is bool:
        return 5
    if type(value) is str:
        return 6 * len(value) + 2
    if type(value) is int:
        return value.bit_length() + 3  # Decimal digits and sign fit within this binary-digit bound.
    if type(value) is float:
        return 32  # Finite IEEE754 repr; full Config/record validation remains mandatory.
    if isinstance(value, Path):
        return 6 * sum(len(part) + 1 for part in value.parts) + 2  # No str()/asdict path copy.
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ContractError('cyclic final validation control structure')
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return 2 + sum(len(field.name) + 4 + _wire_upper(getattr(value, field.name), active) for field in fields(value))
        if type(value) in (list, tuple):
            return 2 + sum(1 + _wire_upper(item, active) for item in value)
        if type(value) is dict:
            return 2 + sum(2 + _wire_upper(key, active) + _wire_upper(item, active) for key, item in value.items())
        raise ContractError('unknown final validation control size type')
    finally:
        active.remove(identity)


def _object_wire(fields: tuple[tuple[str, int], ...]) -> int:
    return 2 + sum(len(key) + 4 + size for key, size in fields)


def _control_bound(loaded: Any, parent: FormalGameParent, output: Path) -> int:
    """Complete clone/project/schedule/registry/signature overlap, before allocation.

    Length arithmetic only: no planned policy, artifact path, JSON or Source
    permission is fabricated. The maxima below are the CLOSED validation recipe:
    32 pairs,64 selected seats, uint64 streams and three registry identities.
    """
    state_bytes, record_bytes = parent.canonical_control_sizes
    loaded_wire = _wire_upper(loaded)
    parent_chars = sum(len(part) + 1 for part in parent.directory.parts)
    output_chars = sum(len(part) + 1 for part in output.parts)
    ancestors = len(parent.directory.parts) + len(output.parts) + 1
    label_chars = len('transformer') + len('-ppo-v3-') + 64 + 1 + 64
    opponent_chars = len('heuristic-card-instance-v2')
    episode_chars, pair_chars = len('validation-p31-seat1'), len('validation-p31')
    episode_wire = _object_wire((('episode_id', episode_chars + 2),
        ('policies', 2 + label_chars + 2 + 1 + opponent_chars + 2),
        ('sampling_seeds', 2 + 20 + 1 + 20), ('pair_id', pair_chars + 2), ('learner_viewer', 1)))
    group_wire = _object_wire((('seed', 36 + 2), ('episodes', 2 + 2 * (episode_wire + 1)), ('ancestry', 4)))
    schedule_wire = 2 + 32 * (group_wire + 1)
    # Safe options are subsets of the protected original state plus the actual
    # parent path/six-digit artifact suffix and fixed own-binding field names.
    binding_wire = 6 * state_bytes + 6 * (parent_chars + len('/artifacts/') + 6) + _wire_upper((
        'version', 'checkpoint', 'expected_manifest_digest', 'model', 'inference', 'cuda_visible_devices',
        'action_config_hash', 'qualified_game', '3'))
    neural_wire = _object_wire((('model_id', len('transformer') + 2), ('policy_options', binding_wire)))
    baseline_wire = _object_wire((('model_id', len('heuristic') + 2), ('policy_options', 4)))
    registry_wire = 2 + label_chars + 3 + neural_wire + 2 * (opponent_chars + 3 + baseline_wire)
    seats_wire = 2 + 64 * (episode_chars + 5)
    linkage_wire = _object_wire((('parent_run_id', 6 * state_bytes + 2),
        ('parent_run_path', 6 * parent_chars + 2), ('batch_id', len('validation-final') + 2),
        ('final_generation', 4), ('final_inference_sha256', 66), ('policy_label', label_chars + 2),
        ('learner_seats', seats_wire), ('schedule_digest', 66), ('policy_registry_digest', 66)))
    projection_wire = _object_wire((('recipe', len('run-split-train-to-validation-v1') + 2),
        ('parent_config_sha256', 66), ('parent_config_inputs_sha256', 66), ('projected_config_sha256', 66)))
    invocation_wire = _object_wire(tuple((key, 32) for key in (
        'started_unix', 'started_monotonic', 'caller_deadline_monotonic', 'admission_deadline_monotonic',
        'execution_deadline_monotonic', 'wall_deadline_monotonic')))
    plan_wire = _object_wire((('groups', schedule_wire), ('bindings', registry_wire),
        ('output', 6 * output_chars + 2), ('linkage', linkage_wire), ('config_projection', projection_wire),
        ('validation_invocation', invocation_wire), ('admission_deadline', 32), ('execution_deadline', 32),
        ('wall_deadline', 32), ('started_monotonic', 32), ('kind', len('ppo_final_validation') + 2)))
    # Eight overlapping object/text/encoding work representations cover parent
    # JSON clones, both projected Configs, asdict/clone(schedule), the plan,
    # canonical signature and encodings. Use the existing32x JSON policy plus
    #64KiB scratch per representation; ancestor Path/control storage is additive.
    return 8 * (32 * (state_bytes + record_bytes + loaded_wire + plan_wire) + _CONTROL_SCRATCH) + (
        ancestors * (_CONTROL_SCRATCH + 32 * (parent_chars + output_chars)))


def _reserve_controls(loaded: Any, parent: FormalGameParent, output: Path, *,
                      extra_owned_bytes: int = 0, max_owned_bytes: int | None = None) -> int:
    # The only allocations before this are fixed-size checks/borrow bookkeeping.
    # Admit sizing scratch before traversing any already-owned control structure.
    parent.admit_external(extra_owned_bytes + _CONTROL_SCRATCH)
    total = extra_owned_bytes + _control_bound(loaded, parent, output)
    parent.admit_external(total)
    if max_owned_bytes is not None and parent.resident_bytes + total > max_owned_bytes:
        raise ContractError('final validation caller control/copy reservation exhausted before materialization')
    return total


def _facts(parent: FormalGameParent, loaded: Any) -> tuple[Json, str, RuntimePolicyBinding]:
    if type(parent) is not FormalGameParent:
        raise ContractError('SAME concrete active original FormalGameParent required')
    parent.require_active()
    parent.recheck()
    state = parent.state
    if (state is None or state['status'] != 'complete' or not 1 <= integer(state['updates'], 1, 1000) <= 1000
            or canonical(state['effective_config']) != canonical(loaded.effective)
            or state['config_sha256'] != loaded.sha256 or canonical(state['config_inputs']) != canonical(list(loaded.inputs))):
        raise ContractError('validation caller Config/input origins differ from original formal parent')
    label, binding = parent.final_binding()
    if type(label) is not str or type(binding) is not RuntimePolicyBinding or binding.policy_options is None:
        raise ContractError('original final retained NN policy binding required')
    return state, label, binding


def _project(loaded: Any) -> Any:
    config = replace(loaded.config, run=replace(loaded.config.run, split='validation'))
    effective = asdict(config)
    return replace(loaded, config=config, effective=effective, sha256=digest(effective))


def _registry(label: str, binding: RuntimePolicyBinding) -> dict[str, RuntimePolicyBinding]:
    return {label: binding, 'random-card-instance-v2': RuntimePolicyBinding('random'),
        'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')}


def _plan(loaded: Any, parent: FormalGameParent, output: Path, invocation: Json) -> GameFinalValidationPlan:
    state, label, binding = _facts(parent, loaded)
    assert binding.policy_options is not None
    projected = _project(loaded)
    groups = validation_groups(state['seed'], label, actions_version=3)
    registry = _registry(label, binding)
    linkage = {'parent_run_id': state['run_id'], 'parent_run_path': str(parent.directory),
        'batch_id': 'validation-final', 'final_generation': state['updates'],
        'final_inference_sha256': binding.policy_options['expected_manifest_digest'], 'policy_label': label,
        'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes},
        'schedule_digest': digest(serialize_schedule(groups, 4)),
        'policy_registry_digest': digest({label: asdict(value) for label, value in registry.items()})}
    return GameFinalValidationPlan(groups, registry, output, linkage,
        config_projection(loaded.effective, list(loaded.inputs), projected.effective), invocation,
        invocation['admission_deadline_monotonic'], invocation['execution_deadline_monotonic'],
        invocation['wall_deadline_monotonic'], invocation['started_monotonic'])


def _signature(plan: GameFinalValidationPlan) -> str:
    return canonical({**asdict(plan), 'output': str(plan.output)})


def _ancestors(path: Path) -> dict[Path, tuple[int, int]]:
    snapshots = {}
    for parent in path.parents:
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ContractError('validation path ancestors must be nonsymlink directories')
        snapshots[parent] = info.st_dev, info.st_ino
    return snapshots


def _check_ancestors(snapshots: dict[Path, tuple[int, int]]) -> None:
    for path, expected in snapshots.items():
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != expected:
            raise ContractError('validation output ancestor changed')


def _output(output: Path, parent: FormalGameParent, *, fresh: bool) -> tuple[Path, dict[Path, tuple[int, int]]]:
    lexical = Path(output).absolute()
    snapshots = _ancestors(lexical)
    resolved = lexical.resolve(strict=False)
    if resolved == parent.directory or resolved.is_relative_to(parent.directory) or parent.directory.is_relative_to(resolved):
        raise ContractError('validation output must be outside original formal parent')
    if fresh and (lexical.exists() or lexical.is_symlink()):
        raise ContractError('fresh validation output already exists')
    if not fresh:
        if not resolved.is_dir() or not stat.S_ISDIR(lexical.lstat().st_mode):
            raise ContractError('completed validation original directory required')
        info = resolved.lstat()
        snapshots[resolved] = info.st_dev, info.st_ino
    return resolved, snapshots


def _actor_ledger(ledger: ReservationLedger, loaded: Any) -> None:
    run = loaded.config.run
    if (type(ledger) is not ReservationLedger or ledger.learner is not None or ledger.learner_bytes
            or ledger.workers or ledger.updating or ledger.cuda_bytes
            or ledger.global_cuda_bytes != run.global_cuda_budget_bytes or ledger.max_workers != run.max_participant_workers):
        raise ContractError('SAME empty actor-only ledger with no learner/lease/update required')


class _IssuedFinalValidationCollection:
    """Actual pending I/O ownership, not Source or Dataset qualification."""
    def __init__(self, context: GameFinalValidationContext, publication: CollectionPublication) -> None:
        self._context: GameFinalValidationContext | None = context
        self.publication = publication
        self._report = b''

    @property
    def directory(self) -> Path:
        return self.publication.directory

    @property
    def report(self) -> Json:
        if not self._report:
            raise ContractError('issued final validation report is not live')
        if self._context is None:
            raise ContractError('final validation collection token has retired')
        self.require(self._context)
        self._context._admit_current()
        return json.loads(self._report)

    def require(self, context: GameFinalValidationContext) -> None:
        if (type(context) is not GameFinalValidationContext or self._context is not context
                or context._issued is not self or type(self.publication) is not CollectionPublication
                or self.publication.directory != context._output or not self.publication.final_pending):
            raise ContractError('SAME actual context-issued pending collection required')
        context._require()

    def manifest_name(self) -> str:
        if self._context is None:
            raise ContractError('final validation collection token has retired')
        self.require(self._context)
        return self.publication.pending_manifest()

    def retire_memory(self) -> None:
        self.publication.retire_final_memory()
        self._report = b''
        self._context = None


class GameFinalValidationSource:
    """Retained completed originals; no update, numeric or Dataset permission."""
    def __init__(self, parent: FormalGameParent, snapshot: _SourceSnapshot | None, selected: tuple[EpisodeRecords, ...] | None,
                 facts: Json, *, ancestors: dict[Path, tuple[int, int]] | None = None,
                 pending: _IssuedFinalValidationCollection | None = None, reservation: int = 0,
                 native: Any = None, budget_loan: FinalSourceBudgetLoan | None = None,
                 manifest_ownership: Any = None) -> None:
        if native is not None:
            from gf_training.native_card_instance import _NativeFinalValidationInputs
            if (snapshot is not None or type(native) is not _NativeFinalValidationInputs
                    or getattr(parent, '_native_final_inputs', None) is not native):
                raise ContractError('mutually exclusive SAME-parent native final64 loan required')
        elif snapshot is None:
            raise ContractError('original Python snapshot or actual native loan required')
        if budget_loan is not None:
            if type(budget_loan) is not FinalSourceBudgetLoan:
                raise ContractError('concrete actual-parent resource loan required')
            budget_loan.require(parent)
        self._original_contract: _Final64OriginalContract | None = None
        self._budget_loan = budget_loan
        self._manifest_ownership = manifest_ownership
        self._parent, self._selected = parent, selected
        self._native = native
        self._management: _SourceOwner | None = None
        self._management_retained = 0
        self._native_facts_reservation = 0
        self._reserved_bytes = reservation
        self._snapshot: _SourceSnapshot | None = snapshot
        self._ancestors = {} if ancestors is None else dict(ancestors)
        self._facts = canonical(facts).encode()
        self._active, self._closed = True, False
        self._borrow_token: object | None = None
        self._pending = pending
        self._failure: BaseException | None = None

    def _require(self) -> None:
        if self._failure is not None:
            raise ContractError('final validation Source permanently poisoned') from self._failure
        if not self._active or self._closed or self._selected is None:
            raise ContractError('SAME active retained final validation Source required')

    @property
    def facts(self) -> Json:
        self.recheck()
        try:
            if self._reserved_bytes:
                _admit_source(self._parent, self._budget_loan, self._reserved_bytes)
            if self._native is not None:
                needed = 32*len(self._facts) + 65536
                proposed = self._reserved_bytes + max(0, needed-self._native_facts_reservation)
                assert self._management is not None
                if self._parent.resident_bytes + proposed > self._management.limit:
                    raise ContractError('native final64 caller facts allowance exhausted before decode')
                _admit_source(self._parent, self._budget_loan, proposed)
                self._reserved_bytes = proposed
                self._native_facts_reservation = needed
                return json.loads(self._facts)
            assert self._snapshot is not None
            self._snapshot.reserve(32 * len(self._facts) + 65536)
            return json.loads(self._facts)
        except BaseException as error:
            self._failure = error
            if self._native is not None:
                self._native._remember(error)
            raise

    @property
    def selected_records(self) -> tuple[EpisodeRecords, ...]:
        self.recheck()
        # All records/bank/frame/table/Selection containers are the shared
        # immutable byte-backed own-participant DTOs, never management controls.
        assert self._selected is not None
        return self._selected

    @contextmanager
    def borrow_original_selected_episode(self, ordinal: int, *, budget_loan: FinalSourceBudgetLoan) -> Iterator[Any]:
        from gf_training.card_instance_source import OriginalGameEpisodeBorrow
        self.recheck()
        if budget_loan is not self._budget_loan or self._parent._unit_borrow is not None:
            raise ContractError('SAME actual Source/resource loan and idle original borrow required')
        budget_loan.require(self._parent)
        index=integer(ordinal,0,63)
        assert self._selected is not None and len(self._selected)==64 and self._original_contract is not None
        # Resolve original identity from the fixed protected final schedule, not
        # a native compact summary or caller-supplied ordinal label.
        budget_loan.admit_local(self._reserved_bytes+32*len(self._original_contract.expected_facts_bytes)+64*4096+65536)
        facts=json.loads(self._original_contract.expected_facts_bytes)
        groups=validation_groups(facts['seed'],facts['final_binding']['label'],actions_version=3)
        spec=tuple(spec for group in groups for spec in group.episodes)[index]
        record=self._selected[index]
        if record.bank.actor!=spec.learner_viewer:
            raise ContractError('original final selected viewer differs')
        budget_loan.admit_local(self._reserved_bytes+65536)
        borrow=OriginalGameEpisodeBorrow._from_owner(self._parent,self,budget_loan,
            episode_id=spec.episode_id,record=record)
        try:
            yield borrow
        except BaseException as error:
            borrow._failure=error
            self._failure=self._failure or error
            self._parent._poison(error)
            raise
        finally:
            del facts,groups,spec,record
            try:
                borrow.close()
            finally:
                if borrow._closed:
                    borrow._finish_scope()

    def _management_guard(self) -> None:
        if self._management is not None:
            from gf_training.group_pack_game_source import PackedSourceOwner
            if type(self._management) is PackedSourceOwner:
                self._management.finalize()
                return
            # Preserve the actual native ownership and all actor/caller charges
            # before any repeated management hash/read scratch is admitted.
            base = 0 if self._budget_loan is None else self._budget_loan.base_bytes
            extra = self._parent.resident_bytes + base + self._reserved_bytes - self._management_retained
            self._management.used += extra - self._management.extra_owned_bytes
            self._management.extra_owned_bytes = extra
            if self._management.used > self._management.limit:
                raise ContractError('final64 management joint ownership exhausted before recheck')
            self._management.finalize()

    def _custody_guard(self) -> None:
        if self._budget_loan is not None:
            self._budget_loan.require(self._parent)
        if self._manifest_ownership is not None:
            self._manifest_ownership.require(self._manifest_ownership.directory)

    def recheck(self) -> None:
        self._require()
        try:
            _check_ancestors(self._ancestors)
            if self._budget_loan is not None:
                self._budget_loan.require(self._parent)
            if self._manifest_ownership is not None:
                self._manifest_ownership.require(self._manifest_ownership.directory)
            if self._pending is not None:
                self._pending.manifest_name()
            self._parent.recheck()
            if self._native is not None:
                self._management_guard()
                self._native.recheck()
            else:
                assert self._snapshot is not None
                self._snapshot.recheck()
            _check_ancestors(self._ancestors)
        except BaseException as error:
            self._failure = error
            if self._native is not None:
                self._native._remember(error)
            raise

    @property
    def resident_bytes(self) -> int:
        self.recheck()
        if self._budget_loan is None:
            raise ContractError('exclusive Source counter requires actual resource-loan Source')
        return self._budget_loan.source_bytes

    @property
    def cleanup_complete(self) -> bool:
        return (self._closed and self._snapshot is None and self._borrow_token is None
            and self._native is None and self._management is None)

    def close(self) -> None:
        if self._closed:
            return
        if self._parent._unit_borrow is not None:
            raise ContractError('actual original episode borrow must retire before Source close')
        if self._native is not None:
            self._close_native()
            return
        failure = self._failure
        if self._snapshot is not None:
            for guard in (self._custody_guard,self._parent.recheck, self._snapshot.recheck,
                    lambda: _check_ancestors(self._ancestors)):
                try:
                    guard()
                except BaseException as error:  # noqa: BLE001 - every independent original guard is reportable
                    previous = failure
                    failure = failure or error
                    self._failure = failure
                    vars(failure)['_game_final_validation_cleanup_owner'] = self
                    vars(failure)['_game_final_validation_parent_owner'] = self._parent
                    if previous is not None and previous is not error:
                        _note(previous, 'final validation Source guard: ', error)
            self._active = False
            self._selected, self._facts = (), b''
            self._original_contract = None
            self._snapshot = None
            self._pending = None
            self._ancestors.clear()
        if self._borrow_token is not None:
            try:
                if self._budget_loan is None:
                    self._parent.admit_external(0)
                _release_borrow(self._parent, self._borrow_token)
                self._borrow_token = None
            except BaseException as error:  # noqa: BLE001 - uncertain borrow retains SAME failed Source owner
                failure = failure or error
        self._closed = self._snapshot is None and self._borrow_token is None
        if self._closed and self._budget_loan is not None:
            try:
                self._budget_loan.retire_local(self)
            except BaseException as accounting:  # noqa: BLE001 - preserve actual accounting cleanup owner
                failure = failure or accounting
                vars(failure)['_game_final_validation_cleanup_owner'] = self
        if failure is not None:
            self._failure = failure
            if not self._closed:
                vars(failure)['_game_final_validation_cleanup_owner'] = self
            raise failure

    def _close_native(self) -> None:
        failure = self._failure
        if failure is not None:
            self._native._remember(failure)
        for guard in (self._custody_guard,self._parent.recheck, self._management_guard,
                lambda: _check_ancestors(self._ancestors)):
            try:
                guard()
            except BaseException as error:  # noqa: BLE001 - every retained original guard is reportable
                previous = failure
                failure = failure or error
                self._failure = failure
                vars(failure)['_game_final_validation_cleanup_owner'] = self
                vars(failure)['_game_final_validation_parent_owner'] = self._parent
                if previous is not None and previous is not error:
                    _note(previous, 'native final64 Source guard: ', error)
        self._active = False
        # Own DTOs retire before the slot/loan; never present fake empty records.
        self._selected, self._facts = None, b''
        self._original_contract = None
        native = self._native
        try:
            native.close()
        except BaseException as error:  # noqa: BLE001 - confirmed retirement can still report the first failure
            previous = failure
            failure = failure or error
            self._failure = failure
            vars(failure)['_game_final_validation_cleanup_owner'] = self
            vars(failure)['_game_final_validation_parent_owner'] = self._parent
            if previous is not None and previous is not error:
                _note(previous, 'native final64 retirement: ', error)
        if not native.cleanup_complete:
            failure = failure or ContractError('native final64 actual slot retirement remains unconfirmed')
            self._failure = failure
            vars(failure)['_game_final_validation_cleanup_owner'] = self
            vars(failure)['_game_final_validation_parent_owner'] = self._parent
            raise failure
        self._native = self._management = None
        self._pending = None
        self._ancestors.clear()
        if self._borrow_token is not None:
            try:
                if self._budget_loan is None:
                    self._parent.admit_external(0)
                _release_borrow(self._parent, self._borrow_token)
                self._borrow_token = None
            except BaseException as error:  # noqa: BLE001 - keep actual exclusion when release is uncertain
                failure = failure or error
        self._closed = self._borrow_token is None
        if self._closed and self._budget_loan is not None:
            try:
                self._budget_loan.retire_local(self)
            except BaseException as accounting:  # noqa: BLE001 - preserve actual accounting cleanup owner
                failure = failure or accounting
                vars(failure)['_game_final_validation_cleanup_owner'] = self
        if failure is not None:
            self._failure = failure
            vars(failure)['_game_final_validation_cleanup_owner'] = self
            vars(failure)['_game_final_validation_parent_owner'] = self._parent
            native._remember(failure)
            raise failure


def _open_original(loaded: Any, parent: FormalGameParent, directory: Path, *, deadline: float,
                   max_owned_bytes: int, expected_plan: GameFinalValidationPlan | None = None,
                   returned: Json | None = None, extra_owned_bytes: int = 0,
                   pending: _IssuedFinalValidationCollection | None = None,
                   recorded_build: Json | None = None) -> GameFinalValidationSource:
    _caller_contract(loaded)
    require_actor_backend(loaded, recorded_build=recorded_build)
    control_bytes = _reserve_controls(loaded, parent, directory, extra_owned_bytes=extra_owned_bytes,
        max_owned_bytes=max_owned_bytes)
    _loaded(loaded, recorded_build=recorded_build)
    state, label, binding = _facts(parent, loaded)
    assert binding.policy_options is not None
    directory, ancestors = _output(directory, parent, fresh=False)
    projected = _project(loaded)
    actual_controls = extra_owned_bytes + _owned_bytes((loaded, projected, state, label, binding, {str(path): value for path, value in ancestors.items()}))
    if actual_controls > control_bytes:
        raise ContractError('final validation materialized controls exceed prior reservation')
    contract = _original_contract(parent, state=state)
    return _inspect_final64_originals(contract, parent, directory, deadline=deadline,
        max_owned_bytes=max_owned_bytes, expected_plan=expected_plan, returned=returned,
        pending=pending, control_bytes=control_bytes, ancestors=ancestors)


def _inspect_final64_originals(contract: _Final64OriginalContract, parent: FormalGameParent, directory: Path, *,
        deadline: float, max_owned_bytes: int, expected_plan: GameFinalValidationPlan | None,
        returned: Json | None, pending: _IssuedFinalValidationCollection | None, control_bytes: int,
        ancestors: dict[Path,tuple[int,int]], budget_loan: FinalSourceBudgetLoan | None = None,
        manifest_ownership: Any = None) -> GameFinalValidationSource:
    if type(contract) is not _Final64OriginalContract or contract.parent is not parent:
        raise ContractError('SAME protected actual-parent original contract required')
    effective=json.loads(contract.effective_bytes)
    origins=json.loads(contract.input_origin_bytes)
    projected=json.loads(contract.projected_effective_bytes)
    expected=json.loads(contract.expected_facts_bytes)
    binding=RuntimePolicyBinding(**expected['final_binding']['binding'])
    assert binding.policy_options is not None
    if contract.settings.backend=='rust':
        return _inspect_native_final64_originals(contract,parent,directory,deadline=deadline,
            max_owned_bytes=max_owned_bytes,expected_plan=expected_plan,returned=returned,pending=pending,
            control_bytes=control_bytes,ancestors=ancestors,budget_loan=budget_loan,manifest_ownership=manifest_ownership)
    packed = 'storage' in effective
    owner: Any = (parent._open_packed_final_originals(directory, contract.settings, deadline=deadline, pending=pending)
        if packed else _SourceOwner(directory, contract.settings, deadline,
        extra_owned_bytes=parent.resident_bytes + control_bytes + (0 if budget_loan is None else budget_loan.base_bytes), max_owned_bytes=max_owned_bytes))
    manifest_name = 'manifest.json' if pending is None else pending.manifest_name()
    collection = _inspect_component_collection(owner, contract.actions, 'validation', _game=True,
        _game_role=FINAL_VALIDATION, _game_pending=pending)
    manifest = collection.manifest
    invocation = checked_invocation(manifest['validation_invocation'], run=effective['run'])
    plan = _plan_original_contract(contract, directory, invocation)
    if expected_plan is not None and _signature(plan) != _signature(expected_plan):
        raise ContractError('completed final validation differs from SAME frozen live plan')
    checked_projection(manifest['config_projection'], parent=effective, inputs=origins, projected=manifest['config'])
    checked_linkage(manifest['linkage'], role=FINAL_VALIDATION, schedule=manifest['schedule'], registry=manifest['policy_bindings'])
    if (canonical(manifest['config']) != canonical(projected) or manifest['config_hash'] != digest(projected)
            or canonical(manifest.get('inputs')) != canonical(origins)
            or canonical(manifest.get('code')) != canonical(expected['code'])
            or canonical(manifest['linkage']) != canonical(plan.linkage)
            or canonical(manifest['schedule']) != canonical(serialize_schedule(plan.groups, 4))
            or canonical(manifest['policy_bindings']) != canonical({key: asdict(value) for key, value in plan.bindings.items()})
            or manifest['status'] != 'complete'):
        raise ContractError('completed final validation original Config/roles/registry/plan differs')
    for receipt in manifest['runtime_evidence']:
        if receipt['binary']['sha256'] != expected['integration']['binary_sha256']:
            raise ContractError('final validation original runtime binary differs from parent')
    report = owner.object(directory / 'report.json')
    duration = invocation['wall_deadline_monotonic'] - invocation['started_monotonic']
    elapsed = manifest['elapsed_seconds']
    if (report.get('healthy') is not True or report.get('planned_games') != 64 or report.get('natural_denominator') != 64
            or report.get('comparison_valid') is not True
            or any(report.get(name) for name in ('failures', 'health_failures', 'group_health_failures'))
            or type(report.get('wall_seconds')) not in (float, int) or not math.isfinite(report['wall_seconds'])
            or not 0 <= report['wall_seconds'] <= elapsed <= duration):
        raise ContractError('all64 healthy natural final validation games and original cost order required')
    if returned is not None:
        _checked_returned_report(report, returned, elapsed, duration)
    seats = plan.linkage['learner_seats']
    selected = tuple(row.records for row in collection.episodes if seats[row.episode_id] == row.viewer)
    if len(selected) != 64 or any(row.end_kind != 'natural' for row in selected):
        raise ContractError('all64 selected own final-policy natural endpoints required')
    snapshot = _SourceSnapshot(collection, owner,
        _owned_bytes((collection.episodes, collection._manifest, collection._proof, owner.files, owner.inventories)))
    snapshot.check()
    _admit_source(parent, budget_loan, control_bytes + snapshot._retained_bytes)
    _check_ancestors(ancestors)
    facts = {
        'directory': str(directory), 'manifest_sha256': owner.files[manifest_name]['sha256'],
        'parent_run_id': expected['parent_run_id'], 'final_generation': expected['final_generation'],
        'final_inference_sha256': binding.policy_options['expected_manifest_digest'],
        'config_projection': manifest['config_projection'], 'validation_invocation': invocation,
        'elapsed_seconds': elapsed, 'games': 64, 'selected_decisions': sum(len(row.frames) for row in selected),
        'zero_endpoint_games': sum(not row.frames for row in selected),
        'qualified': False, 'parent_qualification': 'not_performed'}
    if packed:
        from gf_common.card_instance_collection import packed_final_facts
        facts = packed_final_facts(facts, storage=owner.storage, custody=owner.custody,
            source_set_sha256=collection.proof['source_set_sha256'])
    source = GameFinalValidationSource(parent, snapshot, selected, facts, ancestors=ancestors, pending=pending,
        reservation=control_bytes + snapshot._retained_bytes, budget_loan=budget_loan, manifest_ownership=manifest_ownership)
    if budget_loan is not None:
        budget_loan._bind_source(source)
    try:
        source.recheck()
        return source
    except BaseException as error:
        try:
            source.close()
        except BaseException as cleanup:  # noqa: BLE001 - reportable rechecks cannot hide admission failure
            if not source.cleanup_complete:
                vars(error)['_game_final_validation_cleanup_owner'] = source
            if cleanup is not error:
                _note(error, 'final validation admission retirement: ', cleanup)
        raise


def _open_native_original(loaded: Any, parent: FormalGameParent, directory: Path, *, deadline: float,
        max_owned_bytes: int, expected_plan: GameFinalValidationPlan | None, returned: Json | None,
        pending: _IssuedFinalValidationCollection | None, control_bytes: int, state: Json,
        binding: RuntimePolicyBinding, projected: Any,
        ancestors: dict[Path, tuple[int,int]]) -> GameFinalValidationSource:
    # Compatibility adapter for existing caller-owned Actor path; readonly path
    # never manufactures a Loaded object or enters this adapter.
    contract=_original_contract(parent,state=state)
    return _inspect_native_final64_originals(contract,parent,directory,deadline=deadline,
        max_owned_bytes=max_owned_bytes,expected_plan=expected_plan,returned=returned,pending=pending,
        control_bytes=control_bytes,ancestors=ancestors)


def _inspect_native_final64_originals(contract: _Final64OriginalContract, parent: FormalGameParent, directory: Path, *,
        deadline: float,max_owned_bytes:int,expected_plan:GameFinalValidationPlan|None,returned:Json|None,
        pending:_IssuedFinalValidationCollection|None,control_bytes:int,ancestors:dict[Path,tuple[int,int]],
        budget_loan:FinalSourceBudgetLoan|None=None, manifest_ownership:Any=None)->GameFinalValidationSource:
    effective=json.loads(contract.effective_bytes)
    origins=json.loads(contract.input_origin_bytes)
    projected=json.loads(contract.projected_effective_bytes)
    expected=json.loads(contract.expected_facts_bytes)
    binding=RuntimePolicyBinding(**expected['final_binding']['binding'])
    assert binding.policy_options is not None
    packed = 'storage' in effective
    packed_owner: Any = (parent._open_packed_final_originals(directory, contract.settings,
        deadline=min(deadline, parent.deadline), pending=pending) if packed else None)
    native = parent._borrow_native_final_inputs(deadline=deadline)
    source: GameFinalValidationSource | None = None
    try:
        source = GameFinalValidationSource(parent, None, None, {}, ancestors=ancestors,
            pending=pending, reservation=control_bytes, native=native, budget_loan=budget_loan, manifest_ownership=manifest_ownership)
        native.bind_source(source)
        if budget_loan is not None:
            budget_loan._bind_source(source)
        manifest_name = 'manifest.json' if pending is None else pending.manifest_name()
        if manifest_name != ('manifest.json' if packed or pending is None else 'manifest.json.tmp'):
            raise ContractError('fixed actual final64 manifest variant required')
        path = directory / manifest_name
        before = None if packed else path.lstat()
        if packed:
            assert packed_owner is not None
            manifest_size = packed_owner._entries['manifest.json']['bytes']
        else:
            assert before is not None
            manifest_size = before.st_size
        if ((before is not None and not stat.S_ISREG(before.st_mode))
                or manifest_size > min(64*1024**2, contract.settings.max_source_file_mib*1024**2)):
            raise ContractError('bounded regular final64 management control required')
        # Admit native/Python/control/decode overlap before reading this ONE
        # management object. No Python participant journal/columns/Replay read.
        management_bound = 32*manifest_size + 65536
        _admit_source(parent, budget_loan, control_bytes + management_bound)
        source._reserved_bytes = control_bytes + management_bound
        owner: Any = (packed_owner if packed else _SourceOwner(directory, contract.settings, min(deadline, parent.deadline),
            extra_owned_bytes=parent.resident_bytes + control_bytes + (0 if budget_loan is None else budget_loan.base_bytes), max_owned_bytes=max_owned_bytes))
        source._management = owner  # Actual snapshots reachable BEFORE read/parse.
        manifest = owner.object(path)
        source._management_retained = (0 if packed else owner.used - owner.extra_owned_bytes)
        if source._management_retained > management_bound:
            raise ContractError('final64 management growth exceeds prior original-size admission')
        invocation = checked_invocation(manifest['validation_invocation'], run=effective['run'])
        plan = _plan_original_contract(contract, directory, invocation)
        if expected_plan is not None and _signature(plan) != _signature(expected_plan):
            raise ContractError('native final64 differs from SAME frozen live plan')
        checked_projection(manifest['config_projection'], parent=effective,
            inputs=origins, projected=manifest['config'])
        checked_linkage(manifest['linkage'], role=FINAL_VALIDATION,
            schedule=manifest['schedule'], registry=manifest['policy_bindings'])
        if (canonical(manifest['config']) != canonical(projected)
                or manifest['config_hash'] != digest(projected)
                or canonical(manifest.get('inputs')) != canonical(origins)
                or canonical(manifest.get('code')) != canonical(expected['code'])
                or canonical(manifest['linkage']) != canonical(plan.linkage)
                or canonical(manifest['schedule']) != canonical(serialize_schedule(plan.groups, 4))
                or canonical(manifest['policy_bindings']) != canonical({key: asdict(value) for key, value in plan.bindings.items()})
                or manifest.get('status') != 'complete'):
            raise ContractError('native final64 original controls differ from SAME parent/plan')
        mode: Json = {'kind': 'canonical'}
        if packed:
            mode = {'kind': 'packed-canonical'} if pending is None else {'kind': 'packed-issued-pending', 'evidence': owner.custody}
        elif pending is not None:
            # owner.object used and closed its independent no-follow read FD.
            # Its recorded fstat facts/hash MUST still match the fixed path;
            # never inspect/seek/duplicate/close the publisher write descriptor.
            pending.manifest_name()
            entry = owner.files[manifest_name]
            current = path.lstat()
            identity = [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns]
            if not stat.S_ISREG(current.st_mode) or identity != entry['identity']:
                raise ContractError('pending final64 independent read evidence changed')
            root_stat = directory.lstat()
            mode = {'kind': 'issued-pending', 'evidence': {
                'directory_identity': {'device': root_stat.st_dev, 'inode': root_stat.st_ino},
                'file_identity': {'device': current.st_dev, 'inode': current.st_ino,
                    'mode': current.st_mode, 'size': current.st_size,
                    'mtime_ns': current.st_mtime_ns, 'ctime_ns': current.st_ctime_ns},
                'bytes': entry['bytes'], 'sha256': entry['sha256']}}
        request: Json = {'schema_version': 2 if packed else 1,
            'kind': 'card-instance-packed-final-validation-request' if packed else 'card-instance-final-validation-request',
            'parent_run_sha256': native.parent_run_sha256, 'directory': str(directory), 'manifest': mode,
            'expected_plan': {'linkage': plan.linkage, 'config_projection': plan.config_projection,
                'validation_invocation': invocation, 'schedule': serialize_schedule(plan.groups, 4),
                'policy_bindings': {key: asdict(value) for key, value in plan.bindings.items()}},
            'returned_report': returned}
        packed_slot = parent._packed_final_originals
        if packed:
            assert packed_slot is not None
            request.update(storage_binding=parent.scan._packed_state['storage_binding'],
                artifact_registry_sha256=packed_slot.provider.registry_sha256)
        # Protect further request/asdict/encoding work with the same prior
        # complete Actor bound, plus already-owned management snapshots.
        if _wire_upper(request) > control_bytes + (0 if packed_slot is None else packed_slot._controls):
            raise ContractError('native final64 request exceeds prior complete control admission')
        encoded = canonical(request).encode()
        if len(encoded) > min(64*1024**2, contract.settings.max_record_mib*1024**2):
            raise ContractError('native final64 request exceeds record/source bound')
        summary = native.inspect(encoded)
        if (summary['parent_run_sha256'] != request['parent_run_sha256']
                or summary['directory'] != str(directory)
                or summary['manifest_kind'] != mode['kind']
                or summary['manifest_sha256'] != owner.files[manifest_name]['sha256']):
            raise ContractError('native final64 differs from retained management byte origins')
        for key in ('linkage', 'config_projection', 'validation_invocation'):
            if canonical(summary[key]) != canonical(request['expected_plan'][key]):
                raise ContractError('native final64 original summary/plan differs')
        if canonical(summary['elapsed_seconds']) != canonical(manifest['elapsed_seconds']):
            raise ContractError('native final64 original elapsed cost differs')
        base = source._reserved_bytes + _owned_bytes((summary, encoded, request))
        _admit_source(parent, budget_loan, base)
        source._reserved_bytes = base
        selected = native.records(base_external=base, max_owned_bytes=max_owned_bytes,
            aggregate_window=native.aggregate_window)
        if (len(selected) != 64 or any(type(row) is not EpisodeRecords or row.end_kind != 'natural' for row in selected)
                or sum(len(row.frames) for row in selected) != summary['selected_decisions']
                or sum(not row.frames for row in selected) != summary['zero_endpoint_games']):
            raise ContractError('native final64 genuine original record population differs')
        _admit_source(parent, budget_loan, base + _owned_bytes(selected) + 65536)
        source._reserved_bytes = base + _owned_bytes(selected) + 65536
        source._selected = selected
        facts = {'directory': str(directory), 'manifest_sha256': summary['manifest_sha256'],
            'parent_run_id': expected['parent_run_id'], 'final_generation': expected['final_generation'],
            'final_inference_sha256': binding.policy_options['expected_manifest_digest'],
            'config_projection': summary['config_projection'], 'validation_invocation': summary['validation_invocation'],
            'elapsed_seconds': summary['elapsed_seconds'], 'games': 64,
            'selected_decisions': summary['selected_decisions'], 'zero_endpoint_games': summary['zero_endpoint_games'],
            'qualified': False, 'parent_qualification': 'not_performed'}
        if packed:
            from gf_common.card_instance_collection import packed_final_facts
            facts = packed_final_facts(facts, storage=summary['source_proof']['storage'],
                custody=summary['source_proof']['custody'], source_set_sha256=summary['source_set_sha256'])
        source._facts = canonical(facts).encode()
        source.recheck()
        return source
    except BaseException as error:
        if source is not None:
            source._failure = error
        native._remember(error)
        vars(error)['_game_final_validation_cleanup_owner'] = source or parent
        vars(error)['_game_final_validation_parent_owner'] = parent
        try:
            if source is None:
                native.close()
            else:
                source.close()
        except BaseException as cleanup:  # noqa: BLE001 - SAME Source/parent keeps uncertain native cleanup
            if cleanup is not error:
                _note(error, 'native final64 admission retirement: ', cleanup)
        raise


@contextmanager
def open_parent_game_final_validation(parent: FormalGameParent, directory: Path, *,
        manifest_ownership: Any, budget_loan: FinalSourceBudgetLoan,
        max_owned_bytes: int, deadline: float) -> Iterator[GameFinalValidationSource]:
    """Actual-parent-derived canonical Source; no CLI/defaults/Loaded proxy."""
    from gf_training.card_instance_source_locks import (
        ValidationLocatorOwnership,
        ValidationManifestOwnership,
    )
    if type(parent) is not FormalGameParent or type(budget_loan) is not FinalSourceBudgetLoan:
        raise ContractError('actual formal parent and concrete resource loan required')
    budget_loan.require(parent)
    packed = getattr(parent.scan, '_packed_state', None) is not None
    if (type(manifest_ownership) is not (ValidationLocatorOwnership if packed else ValidationManifestOwnership)
            or budget_loan.custodian.ownership_for(directory) is not manifest_ownership):
        raise ContractError('SAME complete-custody canonical manifest ownership required')
    manifest_ownership.require(directory)
    token = _acquire_borrow(parent)
    source: GameFinalValidationSource | None = None
    primary: BaseException | None = None
    try:
        contract = _original_contract(parent, budget_loan=budget_loan)
        if contract.settings.backend == 'rust':
            from gf_training.native_backend import original_history_reader_settings
            original_history_reader_settings(contract.settings, recorded_build=contract.recorded_build)
        directory, ancestors = _output(directory, parent, fresh=False)
        budget_loan.admit_local(budget_loan.source_bytes + 65536*(len(ancestors)+1))
        source = _inspect_final64_originals(contract, parent, directory,
            deadline=min(deadline, parent.deadline), max_owned_bytes=max_owned_bytes,
            expected_plan=None, returned=None, pending=None, control_bytes=budget_loan.source_bytes,
            ancestors=ancestors, budget_loan=budget_loan, manifest_ownership=manifest_ownership)
        source._borrow_token = token
        source._original_contract = contract
        del contract
        yield source
    except BaseException as error:
        primary = error
        candidate = getattr(error, '_game_final_validation_cleanup_owner', None)
        if source is None and type(candidate) is GameFinalValidationSource and candidate._parent is parent:
            source = candidate
            if not source.cleanup_complete:
                source._borrow_token = token
        if source is not None:
            source._failure = source._failure or error
        parent._poison(error)
        raise
    finally:
        failure = primary
        try:
            if source is None:
                if primary is not None:
                    try:
                        traceback_module.clear_frames(primary.__traceback__)
                    except BaseException:  # noqa: BLE001, S110 - preserve initiating failure/custody
                        pass
                budget_loan._retire_failed_opening()
                _release_borrow(parent, token)
            else:
                source.close()
        except BaseException as cleanup:  # noqa: BLE001 - actual Source/loan/custody stay reachable
            failure = failure or cleanup
            vars(failure)['_game_final_validation_cleanup_owner'] = source or parent
            vars(failure)['_final_source_budget_cleanup_owner'] = budget_loan
            if primary is not None and cleanup is not primary:
                _note(primary, 'readonly final64 retirement: ', cleanup)
        if primary is None and failure is not None:
            raise failure


@contextmanager
def open_game_final_validation(loaded: Any, parent: FormalGameParent, directory: Path, *,
                               deadline: float, max_owned_bytes: int) -> Iterator[GameFinalValidationSource]:
    """Readonly original opener; the SAME validator also accepts live collection."""
    _caller_contract(loaded)
    recorded_build = getattr(parent._native_inputs, '_recorded_build', None)
    require_actor_backend(loaded, recorded_build=recorded_build)
    token = _acquire_borrow(parent)
    source = None
    primary: BaseException | None = None
    try:
        source = _open_original(loaded, parent, directory, deadline=deadline, max_owned_bytes=max_owned_bytes,
            recorded_build=recorded_build)
        source._borrow_token = token
        yield source
    except BaseException as error:
        primary = error
        if source is None and loaded.config.preprocessing.backend == 'rust':
            # Initialization has unwound before yield; retire its completed
            # management/DTO copy frames before releasing caller accounting.
            # Do not clear user-body frames after a successfully yielded Source.
            try:
                traceback_module.clear_frames(error.__traceback__)
            except BaseException:  # noqa: BLE001, S110 - first error/cleanup owner must survive
                pass
        if source is None:
            candidate = getattr(error, '_game_final_validation_cleanup_owner', None)
            if (type(candidate) is GameFinalValidationSource and candidate._parent is parent
                    and not candidate.cleanup_complete):
                source = candidate
                source._borrow_token = token
        if source is not None:
            source._failure = source._failure or error
        raise
    finally:
        failure = primary
        try:
            if source is None:
                parent.admit_external(0)
                _release_borrow(parent, token)
            else:
                source.close()  # Retires buffers BEFORE accounting/exclusion.
        except BaseException as cleanup:  # noqa: BLE001 - preserve original and actual cleanup capability
            failure = failure or cleanup
            if source is not None and not source.cleanup_complete:
                source._failure = source._failure or failure
                vars(failure)['_game_final_validation_cleanup_owner'] = source
            elif source is None:
                vars(failure)['_game_final_validation_cleanup_parent'] = parent
                vars(failure)['_game_final_validation_cleanup_token'] = token
            if primary is not None and cleanup is not primary:
                _note(primary, 'final validation original cleanup: ', cleanup)
        if primary is None and failure is not None:
            raise failure


class GameFinalValidationContext:
    """One concrete actor-only invocation borrowing a SAME active formal parent."""
    def __init__(self, loaded: Any, parent: FormalGameParent, output: Path, ledger: ReservationLedger,
                 *, deadline: float) -> None:
        _caller_contract(loaded)
        require_actor_backend(loaded)
        started_unix, started_monotonic = time.time(), time.monotonic()
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or deadline <= started_monotonic + loaded.config.run.finalization_seconds):
            raise ContractError('caller final validation duration cannot preserve finalization reserve')
        token = _acquire_borrow(parent)
        try:
            reservation = _reserve_controls(loaded, parent, output)
            _loaded(loaded)
            invocation = validation_invocation(started_unix=started_unix, started_monotonic=started_monotonic,
                caller_deadline_monotonic=deadline, run=loaded.effective['run'])
            _facts(parent, loaded)
            _actor_ledger(ledger, loaded)
            self._output, self._ancestors = _output(output, parent, fresh=True)
            self._loaded, self._parent, self._ledger = loaded, parent, ledger
            self._projected = _project(loaded)
            self._plan: GameFinalValidationPlan | None = _plan(loaded, parent, self._output, invocation)
            self._expected, self._projected_bytes = _signature(self.plan), canonical(self._projected.effective).encode()
            # Retain admitted original size ceilings, never new mutable claims.
            self._plan_wire = _wire_upper(self._plan)
            self._loaded_wire = _wire_upper(self._loaded)
            self._projected_wire = _wire_upper(self._projected)
            self._active, self._entered, self._closed = False, False, False
            self._failure: BaseException | None = None
            self._source: GameFinalValidationSource | None = None
            self._issued: _IssuedFinalValidationCollection | None = None
            self._accepted = False
            self._public_facts: Json | None = None
            self._loan_retired = False
            self._borrow_token = token
            self._reserved_bytes = reservation
            if self._control_bytes() > reservation:
                raise ContractError('final validation controls exceed prior complete reservation')
        except BaseException as error:
            vars(self).clear()  # No Source/actor exists; retire partial constructor controls first.
            try:
                parent.admit_external(0)
                _release_borrow(parent, token)
            except BaseException as cleanup:  # noqa: BLE001 - retain original error and resource uncertainty
                vars(error)['_game_final_validation_cleanup_parent'] = parent
                vars(error)['_game_final_validation_cleanup_token'] = token
                _note(error, 'actor64 constructor retirement: ', cleanup)
            raise
    @property
    def ledger(self) -> ReservationLedger:
        self._require()
        return self._ledger

    def _admit_current(self) -> None:
        self._require()
        try:
            # Public nested plan/Loaded dictionaries may have changed since
            # entry. Size-check without copying BEFORE any repeated serializer,
            # projected/report clone, parent materialization or re-admission.
            if (_wire_upper(self._plan) > self._plan_wire
                    or _wire_upper(self._loaded) > self._loaded_wire
                    or _wire_upper(self._projected) > self._projected_wire):
                raise ContractError('final validation controls grew beyond immutable admitted originals')
            if self._source is not None:
                self._reserved_bytes = max(self._reserved_bytes, self._source._reserved_bytes)
            self._parent.admit_external(self._reserved_bytes)
        except BaseException as error:
            self._record_failure(error)
            raise

    def _control_bytes(self) -> int:
        return _owned_bytes((self._loaded, self._projected, self._expected, self._projected_bytes,
            {str(path): value for path, value in self._ancestors.items()})) + 32 * len(self._expected.encode()) + 65536 + (
            0 if self._issued is None else self._issued.publication.resident_bytes + 32 * len(self._issued._report))

    @property
    def artifact_budget(self) -> int:
        self._require()
        if self._issued is not None:
            raise ContractError('actor artifacts already belong to the staged collection')
        cfg = self._loaded.config
        limit = min(cfg.training.max_owned_bytes, cfg.model.resources.max_owned_bytes,
            cfg.preprocessing.memory_budget_mib * 1024**2)
        self._admit_current()
        if self._control_bytes() > self._reserved_bytes:
            raise ContractError('final validation controls exceed prior complete reservation')
        remaining = limit - self._parent.resident_bytes - self._reserved_bytes
        if remaining <= 1024**2:
            raise ContractError('joint retained parent/actor artifact allowance exhausted')
        return remaining

    @property
    def plan(self) -> GameFinalValidationPlan:
        if self._plan is None:
            raise ContractError('final validation control plan has retired')
        return self._plan

    @property
    def projected(self) -> Any:
        if self._projected is None:
            raise ContractError('final validation projected Config has retired')
        self._admit_current()
        return replace(self._projected, effective=json.loads(self._projected_bytes))

    @property
    def source(self) -> GameFinalValidationSource:
        self._require()
        if self._source is None:
            raise ContractError('no completed retained final validation Source')
        return self._source

    def _require(self) -> None:
        if self._failure is not None:
            raise ContractError('final validation invocation permanently poisoned') from self._failure
        if not self._active or self._closed:
            raise ContractError('SAME active final validation invocation required')

    async def __aenter__(self) -> Self:
        if self._entered:
            error = ContractError('final validation invocation has one lifetime; reentry refused')
            self._failure = self._failure or error
            raise error
        self._entered, self._active = True, True
        try:
            self.validate_collection(self.plan, self._ledger, self.projected, deadline=self.plan.execution_deadline)
            return self
        except BaseException as error:
            try:
                await self.aclose()
            except BaseException as cleanup:  # noqa: BLE001 - preserve admission error and reachable cleanup
                if not self.cleanup_complete:
                    vars(error)['_game_final_validation_cleanup_owner'] = self
                if cleanup is not error:
                    _note(error, 'actor64 admission cleanup: ', cleanup)
            raise

    def validate_collection(self, plan: GameFinalValidationPlan, ledger: ReservationLedger,
                            loaded: Any, *, deadline: float) -> None:
        self._require()
        try:
            # Refuse foreign shape/growth before touching even the internal
            # signature or parent's cloned facts. Exact byte equality remains
            # the subsequent authority check; size never authorizes a change.
            if (plan is not self._plan or ledger is not self._ledger
                    or type(loaded) is not type(self._projected)
                    or _wire_upper(loaded) > self._projected_wire):
                raise ContractError('final validation caller controls exceed original admitted shape/size')
            self._admit_current()
            _actor_ledger(ledger, self._loaded)
            _facts(self._parent, self._loaded)
            if (plan is not self._plan or ledger is not self._ledger or _signature(plan) != self._expected
                    or canonical(loaded.effective) != self._projected_bytes.decode()
                    or canonical(asdict(loaded.config)) != self._projected_bytes.decode()
                    or loaded.sha256 != digest(json.loads(self._projected_bytes))
                    or canonical(list(loaded.inputs)) != canonical(list(self._loaded.inputs))):
                raise ContractError('SAME final validation frozen plan/projected Config/ledger required')
            _check_ancestors(self._ancestors)
            if time.monotonic() >= min(deadline, self.plan.wall_deadline):
                raise TimeoutError('original final validation invocation deadline')
        except BaseException as error:
            self._record_failure(error)
            raise

    def begin_collection(self, publication: CollectionPublication, report: Json, manifest: Json) -> _IssuedFinalValidationCollection:
        self.validate_collection(self.plan, self._ledger, self.projected, deadline=self.plan.wall_deadline)
        try:
            if (self._issued is not None or type(publication) is not CollectionPublication
                    or publication.directory != self._output):
                raise ContractError('one SAME final validation publication required')
            # Size existing producer records without serializing them first.
            reservation = self._reserved_bytes + 64 * (_wire_upper(report) + _wire_upper(manifest)) + 262144
            self._parent.admit_external(reservation)
            self._reserved_bytes = reservation
            issued = _IssuedFinalValidationCollection(self, publication)
            self._issued = issued  # Keep actual owner reachable before any publication I/O.
            publication.begin_final_validation(report, manifest)
            returned = {**report, 'wall_seconds': time.monotonic() - self.plan.started_monotonic}
            issued._report = canonical(returned).encode()
            issued.require(self)
            return issued
        except BaseException as error:
            self._record_failure(error)
            raise

    def accept_collection(self, result: _IssuedFinalValidationCollection) -> GameFinalValidationSource:
        self.validate_collection(self.plan, self._ledger, self.projected, deadline=self.plan.wall_deadline)
        try:
            if type(result) is not _IssuedFinalValidationCollection or self._source is not None:
                raise ContractError('one SAME issued final validation original collection required')
            result.require(self)
            cfg = self._loaded.config
            self._source = _open_original(self._loaded, self._parent, self._output,
                deadline=self.plan.wall_deadline - cfg.run.report_reserve_seconds,
                max_owned_bytes=min(cfg.training.max_owned_bytes, cfg.model.resources.max_owned_bytes,
                    cfg.preprocessing.memory_budget_mib * 1024**2), expected_plan=self.plan, returned=result.report,
                extra_owned_bytes=self._reserved_bytes, pending=result)
            self._reserved_bytes = max(self._reserved_bytes, self._source._reserved_bytes)
            self._accepted = True
            if result.publication.group_pack is not None:
                # SAME accepted Source, before its facade/slot/graphs retire.
                result.publication.group_pack.retain_final_facts(self._source.facts)
                self._reserved_bytes = max(self._reserved_bytes,
                    self._source._reserved_bytes + result.publication.resident_bytes)
                self._parent.admit_external(self._reserved_bytes)
            return self._source
        except BaseException as error:
            if self._source is None:
                if self._loaded.config.preprocessing.backend == 'rust':
                    try:
                        traceback_module.clear_frames(error.__traceback__)
                    except BaseException:  # noqa: BLE001, S110 - diagnostic failure cannot replace primary
                        pass
                candidate = getattr(error, '_game_final_validation_cleanup_owner', None)
                if (type(candidate) is GameFinalValidationSource and candidate._parent is self._parent
                        and not candidate.cleanup_complete):
                    self._source = candidate
            self._record_failure(error)
            raise

    def _record_failure(self, error: BaseException) -> None:
        primary = self._failure or error
        self._failure = primary
        # These are actual cleanup capabilities, established BEFORE invalidation
        # or any diagnostic formatting can fail. Neither is serialized authority.
        vars(primary)['_game_final_validation_cleanup_owner'] = self
        vars(primary)['_game_final_validation_parent_owner'] = self._parent
        if self._source is not None:
            self._source._failure = self._source._failure or primary
        if self._issued is not None and not self._closed:
            vars(primary)['_card_instance_collection_cleanup_owner'] = self._issued.publication
            self._issued.publication.fail_final_validation(primary)
        if primary is not error:
            _note(primary, 'final validation later failure: ', error)

    @property
    def loan_cleanup_complete(self) -> bool:
        actors = not (self._ledger.workers or self._ledger.learner is not None or self._ledger.updating)
        invalidation = (self._failure is None or self._issued is None or self._issued.publication.cleanup_complete)
        return self._loan_retired and actors and invalidation

    @property
    def cleanup_complete(self) -> bool:
        return self._closed and self.loan_cleanup_complete

    async def _retire_loan(self) -> None:
        if self._loan_retired:
            return
        if self._issued is not None and not self._accepted and self._failure is None:
            self._record_failure(ContractError('final validation has no accepted original Source'))
        if self._failure is not None:
            self._record_failure(self._failure)  # Invalidate FIRST, before subsequent cleanup guards.
        for guard in (lambda: _actor_ledger(self._ledger, self._loaded), self._parent.recheck):
            try:
                guard()
            except BaseException as error:  # noqa: BLE001 - independently reportable guards keep first failure
                self._record_failure(error)
        if self._source is not None:
            try:
                self._source.close()
            except BaseException as error:  # noqa: BLE001 - source retirement cannot replace initiating failure
                self._record_failure(error)
        self._active = False
        actors = self._ledger.workers or self._ledger.learner is not None or self._ledger.updating
        source = self._source is None or self._source.cleanup_complete
        if not actors and source:
            self._source = None
            self._expected, self._projected_bytes = '', b''
            self._ancestors.clear()
            self._plan, self._projected, self._loaded = None, None, None
            self._loan_retired = True
        if self._failure is not None:
            if not self.loan_cleanup_complete:
                vars(self._failure)['_game_final_validation_cleanup_owner'] = self
            raise self._failure

    def _finish_retired(self) -> None:
        if self._closed:
            if self._failure is not None:
                raise self._failure
            return
        if not self._loan_retired:
            raise ContractError('confirmed actor/Source/control retirement required before publication')
        if self._issued is not None:
            publication = self._issued.publication
            if self._failure is None:
                try:
                    self._public_facts = (publication.finish_final_validation(context=self)
                        if publication.group_pack is not None else publication.finish_final_validation())
                except BaseException as error:  # noqa: BLE001 - preserve initiating terminal error and owner
                    self._record_failure(error)
            else:
                publication.fail_final_validation(self._failure)
            if not publication.cleanup_complete:
                assert self._failure is not None
                vars(self._failure)['_game_final_validation_cleanup_owner'] = self
                raise self._failure
            self._issued.retire_memory()
        try:
            self._parent.admit_external(0)  # Cleanup-only; no post-publication filesystem work.
            _release_borrow(self._parent, self._borrow_token)
            self._closed = True
        except BaseException as error:  # noqa: BLE001 - retain actual owner on uncertain logical retirement
            self._record_failure(error)
        if self._failure is not None:
            if not self._closed:
                vars(self._failure)['_game_final_validation_cleanup_owner'] = self
            raise self._failure

    async def _retire_for_owned_parent(self, parent: FormalGameParent) -> None:
        if type(parent) is not FormalGameParent or parent is not self._parent:
            error = ContractError('SAME concrete CLI-owned formal parent required for terminal handoff')
            self._record_failure(error)
            raise error
        await self._retire_loan()  # Publisher/exclusion remain owned across actual parent release.

    def _finish_after_owned_parent(self, parent: FormalGameParent) -> Json | None:
        if parent is not self._parent or not parent.cleanup_complete:
            error = ContractError('SAME owned formal-parent cleanup must complete before final exposure')
            self._record_failure(error)
            raise error
        try:
            self._finish_retired()
        except BaseException as error:
            self._record_failure(error)
            raise
        return self._public_facts

    async def aclose(self) -> None:
        if self._closed:
            return
        try:
            await self._retire_loan()
        except BaseException:
            if not self.loan_cleanup_complete:
                raise
        try:
            self._finish_retired()  # Borrowed-parent path closes only its loan, never caller's parent.
        except BaseException as error:
            self._record_failure(error)
            raise

    async def __aexit__(self, kind: object, value: object, traceback: object) -> None:
        original = value if isinstance(value, BaseException) else None
        if original is not None:
            self._record_failure(original)
        elif self._source is None:
            self._record_failure(ContractError('final validation invocation has no completed original Source'))
        try:
            await self.aclose()
        except BaseException as cleanup:
            if original is None:
                raise
            vars(original)['_game_final_validation_cleanup_owner'] = self
            if cleanup is not original:
                _note(original, 'final validation cleanup: ', cleanup)
