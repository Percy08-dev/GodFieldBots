"""Same-host CUDA storage accounting, not allocation or namespace authority.

Backend allowances, enumerated live storage and transient estimates are separate.
Callers must already hold actual admission before constructing any CUDA tensor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from math import prod
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer
from gf_models.rust_player_cuda import NativeMlpRowCudaProfile
from gf_models.structured_config import parameter_shapes
from gf_models.torch_runtime import serialized_torch

StorageKey = tuple[str, int, int, int]


def _cuda_storage(tensor: Any) -> dict[StorageKey, int]:
    device = getattr(tensor, 'device', None)
    if device is None or device.type != 'cuda':
        return {}
    storage = tensor.untyped_storage()
    size = storage.nbytes()
    return {(device.type, device.index, storage.data_ptr(), size): size} if size else {}


def cuda_learner_storage(network: Any, optimizer: Any) -> tuple[dict[StorageKey, int], int]:
    """Observe actual storage at transitions; CPU steps/RNG add no CUDA bytes."""
    storage: dict[StorageKey, int] = {}
    for parameter in network.parameters():
        storage.update(_cuda_storage(parameter))
        storage.update(_cuda_storage(parameter.grad))
    for buffer in network.buffers():
        storage.update(_cuda_storage(buffer))
    largest = 0
    for state in optimizer.state.values() if optimizer is not None else ():
        for name, value in state.items():
            actual = _cuda_storage(value)
            storage.update(actual)
            if actual and name in ('exp_avg', 'exp_avg_sq'):
                largest = max(largest, value.numel())
    return storage, largest


@dataclass(frozen=True)
class CudaCheckpointProfile:
    """Compile validated Model3/4 geometry once, independently of CUDA init."""

    model: Any
    binding: Any
    layout: Any = field(kw_only=True)
    parameter_bytes: int = field(init=False)
    position_bytes: int = field(init=False)
    row: NativeMlpRowCudaProfile = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shapes = parameter_shapes(self.model, layout=self.layout)
        object.__setattr__(self, 'parameter_bytes', 4 * sum(prod(shape) for shape in shapes.values()))
        object.__setattr__(self, 'position_bytes', 4 * self.model.temporal.max_decisions * self.model.temporal.hidden_size)
        object.__setattr__(self, 'row', NativeMlpRowCudaProfile(self.model, self.binding, layout=self.layout))

    @property
    def network_bytes(self) -> int:
        return self.parameter_bytes + self.position_bytes

    @property
    def fixed_bytes(self) -> int:
        return self.model.resources.cuda_context_bytes + self.model.resources.workspace_bytes


@dataclass(frozen=True)
class DeviceUpdatePlan:
    operation_id: str
    generation: int
    learner_device: str
    serving_device: str
    global_bytes: int
    logical_bytes: int
    device_bytes: int
    host_fixed_bytes: int
    learner_fixed_bytes: int
    retained_cache_tensor_bytes: int
    post_update_learner_tensor_bytes: int
    publication_network_bytes: int
    schema_version: int = field(default=1, init=False)
    kind: str = field(default='rust-player-cuda-update-plan-v1', init=False)
    recipe: str = field(default='same-host-phase-storage-v1', init=False)

    def record(self) -> Json:
        return asdict(self)


_PLAN_FIELDS = frozenset(item.name for item in fields(DeviceUpdatePlan))
_BYTE_FIELDS = ('global_bytes', 'logical_bytes', 'device_bytes', 'host_fixed_bytes',
    'learner_fixed_bytes', 'retained_cache_tensor_bytes', 'post_update_learner_tensor_bytes',
    'publication_network_bytes')


def validate_device_plan(value: Any, *, operation_id: str | None = None,
                         generation: int | None = None) -> Json:
    """Closed DATA only. Readers additionally bind actual config/source receipts."""
    raw = closed(value, set(_PLAN_FIELDS))
    integer(raw['schema_version'], 1, 1)
    integer(raw['generation'], 1, 2**63 - 1)
    if (raw['kind'] != 'rust-player-cuda-update-plan-v1'
            or raw['recipe'] != 'same-host-phase-storage-v1'
            or type(raw['operation_id']) is not str or not raw['operation_id'] or '\0' in raw['operation_id']
            or operation_id is not None and raw['operation_id'] != operation_id
            or generation is not None and raw['generation'] != generation):
        raise ContractError('device plan recipe/actual source operation/generation differs')
    for name in _BYTE_FIELDS:
        integer(raw[name], 0, 2**63 - 1)
    integer(raw['global_bytes'], 1, 2**63 - 1)
    integer(raw['logical_bytes'], 1, 2**63 - 1)
    if (raw['learner_device'] not in ('cpu', 'cuda:0') or raw['serving_device'] not in ('cpu', 'cuda:0')
            or raw['learner_device'] == raw['serving_device'] == 'cpu'):
        raise ContractError('device plan requires an actual CUDA role')
    if raw['learner_device'] == 'cpu':
        if any(raw[name] for name in ('device_bytes', 'learner_fixed_bytes', 'post_update_learner_tensor_bytes')):
            raise ContractError('CPU numerical role cannot carry CUDA learner bytes')
    else:
        expected = min(raw['logical_bytes'], raw['global_bytes'] - raw['retained_cache_tensor_bytes']
            - (raw['host_fixed_bytes'] - raw['learner_fixed_bytes']))
        if raw['device_bytes'] <= 0 or raw['device_bytes'] != expected:
            raise ContractError('immutable device allowance differs from phase recipe')
    if (raw['host_fixed_bytes'] < raw['learner_fixed_bytes']
            or raw['serving_device'] == 'cpu' and (raw['retained_cache_tensor_bytes'] or raw['publication_network_bytes'])
            or raw['host_fixed_bytes'] + raw['retained_cache_tensor_bytes']
                + raw['post_update_learner_tensor_bytes'] + raw['publication_network_bytes'] > raw['global_bytes']):
        raise ContractError('device plan mixed-role/publication phase is inconsistent')
    return raw


def bind_update_device_plan(update: Json, *, operation_id: str, learner_device: str,
                             serving_device: str, global_bytes: int) -> Json | None:
    """DATA binding to actual source invocation receipts, never the resume job."""
    if learner_device == serving_device == 'cpu':
        if 'device_plan' in update:
            raise ContractError('allCPU update must omit device_plan')
        return None
    if 'device_plan' not in update:
        raise ContractError('CUDA host update requires embedded device_plan')
    raw = validate_device_plan(update['device_plan'], operation_id=operation_id, generation=update['generation'])
    if (raw['learner_device'] != learner_device or raw['serving_device'] != serving_device
            or raw['global_bytes'] != global_bytes):
        raise ContractError('device plan differs from actual source roles/admitted budget')
    return raw


class DeviceOperation:
    """An operation ceiling is not a reservation of that many physical bytes."""

    def __init__(self, account: RustPlayerDeviceAccount, phase: str, *,
                 owner: Any = None, plan: DeviceUpdatePlan | None = None,
                 storage_bytes: int = 0, required_bytes: int = 0) -> None:
        self.account, self.phase, self.owner = account, phase, owner
        self.plan, self.storage_bytes = plan, storage_bytes
        self.required_bytes = required_bytes
        self.state = 'ACTIVE'
        self.error: BaseException | None = None
        self.references: Any = None
        self._failure_traceback: Any = None
        self._failed_frame_references: list[dict[str, Any]] = []
        self._failed_thread: Any = None
        self._failure_tracebacks: list[Any] = []
        self._failure_errors: list[BaseException] = []
        self._failure_cleanup_owners: list[Any] = []
        self._failure_frame_ids: set[int] = set()
        self.retirement_errors: list[BaseException] = []

    @property
    def device_bytes(self) -> int | None:
        return self.plan.device_bytes if self.plan is not None and self.plan.learner_device != 'cpu' else None

    def complete(self) -> None:
        with serialized_torch():
            if self.account._active is not self or self.state != 'ACTIVE':
                raise ContractError('actual active device operation required')
            if self.phase == 'row' and self.references is not None:
                raise ContractError('GPU row inputs must positively retire inside the Torch lock')
            self.references = self.owner = None
            self.state, self.account._active = 'SETTLED', None

    def fail(self, error: BaseException, *, references: Any = None) -> None:
        import threading
        with serialized_torch():
            if self.account._active is not self or self.state != 'ACTIVE':
                raise ContractError('only the actual active operation can fail once')
            self.error, self.references = error, references
            self._failed_thread = threading.current_thread()
            self._failure_traceback = error.__traceback__
            self._retain_exception_chain(error)
            self.state, self.account._poisoned = 'FAILED_HELD', True

    def _retain_exception_chain(self, error: BaseException) -> None:
        """Only actual cause/context links and their known numerical owners."""
        pending = [error]
        seen_errors = {id(item) for item in self._failure_errors}
        seen_frames = self._failure_frame_ids
        seen_owners = {id(item) for item in self._failure_cleanup_owners}
        while pending:
            current = pending.pop()
            if id(current) in seen_errors:
                continue
            seen_errors.add(id(current))
            self._failure_errors.append(current)
            for linked in (current.__cause__, current.__context__):
                if linked is not None:
                    pending.append(linked)
            child = vars(current).get('cleanup_owner')
            if child is not None and child is not self.owner and id(child) not in seen_owners:
                seen_owners.add(id(child))
                self._failure_cleanup_owners.append(child)
            frame = current.__traceback__
            self._failure_tracebacks.append(frame)
            while frame is not None:
                if id(frame.tb_frame) not in seen_frames:
                    seen_frames.add(id(frame.tb_frame))
                    self._failed_frame_references.append(dict(frame.tb_frame.f_locals))
                frame = frame.tb_next

    def retire_failed(self, *, gpu_settled: bool) -> None:
        """Caller proves its actual fence first; stopped aliases/nested closes follow.

        This data bookkeeping flag is not GPU allocation or termination authority.
        The account stays poisoned after cleanup; no failed parent can resume work.
        """
        import sys
        import threading
        import traceback

        with serialized_torch():
            if self.state == 'FAILED_RETIRED':
                return
            if self.state != 'FAILED_HELD' or not gpu_settled or self.error is None:
                raise ContractError('failed operation needs positive GPU settlement')
            if self._failed_thread is not threading.current_thread() and self._failed_thread.is_alive():
                raise ContractError('failed operation worker is still running')
            active: set[int] = set()
            frame: Any = sys._getframe()
            while frame is not None:
                active.add(id(frame))
                frame = frame.f_back
            for stopped in self._failure_tracebacks:
                while stopped is not None:
                    if id(stopped.tb_frame) in active:
                        raise ContractError('failed operation frames are still executing')
                    stopped = stopped.tb_next
            for stopped in self._failure_tracebacks:
                traceback.clear_frames(stopped)
            for values in self._failed_frame_references:
                values.clear()
            self._failed_frame_references.clear()
            if isinstance(self.references, dict):
                self.references.clear()
            self.references = None
            for child in self._failure_cleanup_owners:
                try:
                    child.close()
                except BaseException as cleanup:
                    mark = len(self._failure_cleanup_owners)
                    self.retirement_errors.append(cleanup)
                    self._retain_exception_chain(cleanup)
                    # A newly reported inner owner retires before retrying its
                    # outer close. Capture BEFORE callers can overwrite fields.
                    self._failure_cleanup_owners[:] = (self._failure_cleanup_owners[mark:]
                        + self._failure_cleanup_owners[:mark])
                    raise
            self.references = self.owner = None
            self._failure_traceback = None
            self._failure_tracebacks.clear()
            self._failure_cleanup_owners.clear()
            self._failure_frame_ids.clear()
            self._failed_thread = None
            self.state, self.account._active = 'FAILED_RETIRED', None


class RustPlayerDeviceAccount:
    """One admitted host's actual CUDA role/profile/storage state under one lock."""

    def __init__(self, *, global_bytes: int, learner_device: str, serving_device: str,
                 learner_profile: CudaCheckpointProfile,
                 admitted_profiles: tuple[CudaCheckpointProfile, ...]) -> None:
        integer(global_bytes, 1, 2**63 - 1)
        if (learner_device not in ('cpu', 'cuda:0') or serving_device not in ('cpu', 'cuda:0')
                or learner_device == serving_device == 'cpu' or not admitted_profiles):
            raise ContractError('actual CUDA host roles/profiles required; allCPU has no account')
        self.global_bytes, self.learner_device, self.serving_device = global_bytes, learner_device, serving_device
        self.learner_profile = learner_profile
        self.context_bytes = max(profile.model.resources.cuda_context_bytes for profile in admitted_profiles)
        self.workspace_bytes = max(profile.model.resources.workspace_bytes for profile in admitted_profiles)
        self.host_fixed_bytes = self.context_bytes + self.workspace_bytes
        self._learner: dict[StorageKey, int] = {}
        self._moment_max_numel = 0
        self._cache: dict[str, dict[StorageKey, int]] = {}
        self._profiles: dict[str, CudaCheckpointProfile] = {}
        self._cache_owners: dict[str, Any] = {}
        self._learner_bytes = self._cache_bytes = 0
        self._active: DeviceOperation | None = None
        self._poisoned = False

    def _healthy(self) -> None:
        if self._poisoned:
            raise ContractError('device account failed; actual operation remains held')

    def _idle(self) -> None:
        self._healthy()
        if self._active is not None:
            raise ContractError('another device operation remains live')

    @property
    def learner_tensor_bytes(self) -> int:
        return self._learner_bytes

    @property
    def retained_cache_tensor_bytes(self) -> int:
        return self._cache_bytes

    def _recount_storage(self) -> None:
        # Transition-only; serving a row reads O(1) totals, never parameter maps.
        storage = {key: size for values in self._cache.values() for key, size in values.items()
            if key not in self._learner}
        self._cache_bytes, self._learner_bytes = sum(storage.values()), sum(self._learner.values())

    def observe_learner(self, storage: dict[StorageKey, int], *, moment_max_numel: int = 0) -> None:
        """DATA observation at init/restore/update, never parameter scans per row."""
        with serialized_torch():
            self._healthy()
            if self.learner_device == 'cpu' and storage:
                raise ContractError('CPU learner cannot contribute CUDA storage')
            self._learner = dict(storage)
            self._moment_max_numel = moment_max_numel
            self._recount_storage()

    def refresh_learner(self, network: Any, optimizer: Any) -> None:
        with serialized_torch():
            storage, largest = cuda_learner_storage(network, optimizer)
            self.observe_learner(storage, moment_max_numel=largest)

    def observe_cache(self, checkpoint: str, profile: CudaCheckpointProfile,
                      storage: dict[StorageKey, int], *, owner: Any = None) -> None:
        with serialized_torch():
            self._healthy()
            if (profile.model.resources.cuda_context_bytes > self.context_bytes
                    or profile.model.resources.workspace_bytes > self.workspace_bytes):
                raise ContractError('GPU checkpoint profile was not bound before host initialization')
            if self.serving_device == 'cpu' and storage:
                raise ContractError('CPU serving cannot contribute CUDA storage')
            self._cache[checkpoint], self._profiles[checkpoint] = dict(storage), profile
            self._cache_owners[checkpoint] = owner
            self._recount_storage()

    def refresh_cache(self, checkpoint: str, profile: CudaCheckpointProfile, network: Any) -> None:
        with serialized_torch():
            storage, _ = cuda_learner_storage(network, None)
            self.observe_cache(checkpoint, profile, storage, owner=network)

    def cached_profile(self, checkpoint: str) -> CudaCheckpointProfile | None:
        with serialized_torch():
            return self._profiles.get(checkpoint)

    def begin_update(self, *, logical_bytes: int, operation_id: str,
                     generation: int, owner: Any = None) -> DeviceOperation:
        with serialized_torch():
            self._idle()
            integer(logical_bytes, 1, 2**63 - 1)
            integer(generation, 1, 2**63 - 1)
            if not operation_id:
                raise ContractError('actual update operation identity required')
            profile, cache = self.learner_profile, self.retained_cache_tensor_bytes
            fixed = profile.fixed_bytes if self.learner_device != 'cpu' else 0
            post = 4 * profile.parameter_bytes + profile.position_bytes if self.learner_device != 'cpu' else 0
            publication = profile.network_bytes if self.serving_device != 'cpu' else 0
            required = self.host_fixed_bytes + max(post, self.learner_tensor_bytes) + cache + publication
            if required > self.global_bytes:
                raise ContractError(f'CUDA publication phase exceeds allowance before update: required={required} limit={self.global_bytes}')
            device = (min(logical_bytes, self.global_bytes - cache - (self.host_fixed_bytes - fixed))
                if self.learner_device != 'cpu' else 0)
            if self.learner_device != 'cpu' and device <= 0:
                raise ContractError('CUDA update has no device allowance')
            plan = DeviceUpdatePlan(operation_id, generation, self.learner_device, self.serving_device,
                self.global_bytes, logical_bytes, device, self.host_fixed_bytes, fixed, cache, post, publication)
            self._active = DeviceOperation(self, 'update', owner=owner, plan=plan)
            return self._active

    def numerical_limit(self, logical_bytes: int) -> int | None:
        """Same fixed-component replacement; CPU numerics receive no CUDA cap."""
        with serialized_torch():
            self._healthy()
            if self.learner_device == 'cpu':
                return None
            limit = min(logical_bytes, self.global_bytes - self.retained_cache_tensor_bytes
                - (self.host_fixed_bytes - self.learner_profile.fixed_bytes))
            if limit <= 0:
                raise ContractError('no same-host numerical device allowance')
            return limit

    def begin_construction(self, checkpoint: str, profile: CudaCheckpointProfile,
                           *, owner: Any = None) -> DeviceOperation:
        with serialized_torch():
            self._idle()
            if (profile.model.resources.cuda_context_bytes > self.context_bytes
                    or profile.model.resources.workspace_bytes > self.workspace_bytes):
                raise ContractError('construction profile was not bound before CUDA initialization')
            extra = profile.network_bytes if checkpoint not in self._cache else 0
            required = self.host_fixed_bytes + self.learner_tensor_bytes + self.retained_cache_tensor_bytes + extra
            if required > self.global_bytes:
                raise ContractError(f'CUDA cache construction exceeds allowance: required={required} limit={self.global_bytes}')
            self._active = DeviceOperation(self, 'construction', owner=owner, storage_bytes=extra)
            return self._active

    def evict_cache(self, checkpoint: str, *, gpu_settled: bool) -> None:
        with serialized_torch():
            self._idle()
            if not gpu_settled:
                raise ContractError('cache credit requires actual removal and GPU settlement')
            del self._cache[checkpoint], self._profiles[checkpoint]
            self._cache_owners.pop(checkpoint, None)
            self._recount_storage()

    def discard_failed_cache(self, checkpoint: str) -> None:
        """After actual provisional removal/fence; poison still forbids new work."""
        with serialized_torch():
            if not self._poisoned:
                raise ContractError('failed cache retirement requires the failed account')
            self._cache.pop(checkpoint, None)
            self._profiles.pop(checkpoint, None)
            self._cache_owners.pop(checkpoint, None)
            self._recount_storage()

    def refresh_failed_learner(self, network: Any, optimizer: Any, *, owner: Any) -> None:
        with serialized_torch():
            if self._active is None or self._active.owner is not owner or self._active.state != 'FAILED_HELD':
                raise ContractError('actual failed learner owner required')
            self._learner, self._moment_max_numel = cuda_learner_storage(network, optimizer)
            self._recount_storage()

    def begin_row(self, checkpoint: str, *, interval_events: int, candidates: int,
                  input_leaf_bytes: int, owner: Any = None) -> DeviceOperation:
        with serialized_torch():
            self._idle()
            estimate = self._profiles[checkpoint].row.estimate(interval_events=interval_events,
                candidates=candidates, input_leaf_bytes=input_leaf_bytes)
            required = (self.host_fixed_bytes + self.learner_tensor_bytes
                + self.retained_cache_tensor_bytes + estimate.peak_bytes)
            if required > self.global_bytes:
                raise ContractError(f'CUDA actual row exceeds allowance: required={required} limit={self.global_bytes}')
            self._active = DeviceOperation(self, 'row', owner=owner, storage_bytes=estimate.peak_bytes)
            return self._active

    def _begin_required(self, phase: str, required: int, owner: Any) -> DeviceOperation:
        self._idle()
        if required > self.global_bytes:
            raise ContractError(f'CUDA {phase} phase exceeds allowance: required={required} limit={self.global_bytes}')
        self._active = DeviceOperation(self, phase, owner=owner, required_bytes=required)
        return self._active

    def begin_initialize(self, *, owner: Any = None) -> DeviceOperation:
        with serialized_torch():
            extra = self.learner_profile.network_bytes if self.learner_device != 'cpu' else 0
            return self._begin_required('initialize', self.host_fixed_bytes + self.retained_cache_tensor_bytes + extra, owner)

    def begin_backend(self, *, owner: Any = None) -> DeviceOperation:
        with serialized_torch():
            return self._begin_required('serving-initialize', self.host_fixed_bytes
                + self.learner_tensor_bytes + self.retained_cache_tensor_bytes, owner)

    def begin_restore(self, *, moment_bytes: int, owner: Any = None) -> DeviceOperation:
        with serialized_torch():
            profile = self.learner_profile
            integer(moment_bytes, 0, 2 * profile.parameter_bytes)
            required = self.host_fixed_bytes + self.retained_cache_tensor_bytes
            if self.learner_device != 'cpu':
                if profile.model.version == '3':
                    required += self.learner_tensor_bytes + profile.network_bytes + moment_bytes
                else:
                    from gf_models.learner_cuda import learner_cuda_minima
                    minima = learner_cuda_minima(profile.model, profile.binding, layout=profile.layout)
                    # Existing fresh restore includes one old P+Q network; use
                    # the actual old owner for a nonfresh call, not another copy.
                    required += (self.learner_tensor_bytes + minima['phases']['restore']
                        - profile.fixed_bytes - profile.network_bytes)
            return self._begin_required('restore', required, owner)

    def begin_export(self, *, owner: Any = None) -> DeviceOperation:
        with serialized_torch():
            profile = self.learner_profile
            required = self.host_fixed_bytes + self.retained_cache_tensor_bytes
            if self.learner_device != 'cpu':
                if profile.model.version == '3':
                    required += self.learner_tensor_bytes + 5 * self._moment_max_numel
                else:
                    from gf_models.learner_cuda import learner_cuda_minima
                    minima = learner_cuda_minima(profile.model, profile.binding, layout=profile.layout)
                    required += minima['phases']['export'] - profile.fixed_bytes
            return self._begin_required('export', required, owner)
