"""Ordinary public launch, CONFIG planning and fixed private HPO operations.

Configuration previews are DATA, not admission or a physical-capacity promise.
Torch/native/process imports belong only to explicit preparation.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gf_cli.config import LoadedConfig
from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.hpo_storage import check_staged_collection_destinations


def _config(loaded: LoadedConfig) -> tuple[Any, bool, float]:
    if (type(loaded) is not LoadedConfig or canonical(asdict(loaded.config)) != canonical(loaded.effective)
            or digest(loaded.effective) != loaded.sha256):
        raise ContractError('actual unchanged LoadedConfig required')
    cfg = loaded.config
    hpo = hasattr(cfg, 'hpo_spec') and cfg.hpo_spec.get('schema_version') == 6
    if not hpo and not hasattr(cfg.run, 'player_runtime'):
        raise ContractError('explicit Rust player configuration required')
    if not hasattr(cfg.run, 'post_host_seconds'):
        raise ContractError('public execution requires explicit post_host_seconds')
    return cfg, hpo, cfg.run.post_host_seconds


def check_native_collection_storage(storage: Any, directory: Path, *, catalog_root: Path | None = None) -> None:
    work = getattr(storage, 'collection_work_root', None)
    if work is not None:
        check_staged_collection_destinations(
            Path(work), directory, catalog_root=catalog_root or Path(storage.db_root))


def startup_preview(loaded: LoadedConfig) -> Json:
    """Known fresh network demand only; no Torch, hardware or live account."""
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    cfg, hpo, _p = _config(loaded)
    row = _startup_geometry(cfg.model.architecture(layout=layout), cfg.model.inference,
        learner=cfg.training.device, global_bytes=cfg.hpo_spec['operating']['managed_cuda_bytes'] if hpo else cfg.run.global_cuda_budget_bytes)
    return {**row,
        'declared_ram_bytes': cfg.hpo_spec['operating']['managed_ram_bytes'] if hpo else 24 * 1024**3,
        'declared_source_bytes': cfg.preprocessing.memory_budget_mib * 1024**2,
        'declared_storage_bytes': cfg.storage.settings().budget_bytes}


def _startup_geometry(model: Any, inference: Any, *, learner: str, global_bytes: int) -> Json:
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.rust_player_device import CudaCheckpointProfile
    profile = CudaCheckpointProfile(model, inference, layout=layout)
    serving = inference.device
    active = learner != 'cpu' or serving != 'cpu'
    global_bytes = global_bytes if active else 0
    fixed = profile.fixed_bytes if active else 0
    learner_bytes = profile.network_bytes if learner != 'cpu' else 0
    serving_bytes = profile.network_bytes if serving != 'cpu' else 0
    startup = fixed + learner_bytes + serving_bytes
    if startup > global_bytes:
        raise ContractError('known fresh same-host network demand exceeds declared CUDA allowance')
    return {'learner_device': learner, 'serving_device': serving,
        'parameter_bytes': profile.parameter_bytes, 'position_bytes': profile.position_bytes,
        'host_fixed_bytes': fixed, 'learner_tensor_bytes': learner_bytes,
        'g0_serving_tensor_bytes': serving_bytes, 'startup_cuda_bytes': startup,
        'declared_cuda_bytes': global_bytes,
        'host_admission': False, 'future_phase_checks_required': True}


def campaign_startup_preview(loaded: Any, *, settings: Any = None) -> Json:
    """Native6 proposal/validation DATA bound to the original campaign CONFIG."""
    from gf_cli.hpo_config import LoadedHPOConfig, RustPlayerHPOCampaignConfig, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings
    if type(loaded) is not LoadedHPOConfig:
        raise ContractError('actual loaded native6 campaign CONFIG required for materialization')
    config = loaded.config
    if (not isinstance(config, RustPlayerHPOCampaignConfig) or not hasattr(config.operating, 'post_host_seconds')
            or canonical(asdict(config)) != canonical(loaded.effective)
            or loaded.sha256 != digest({'campaign': loaded.effective, 'base': loaded.base.sha256})):
        raise ContractError('explicit timed native6 campaign configuration required')
    selected = effective_settings(config.model, config.training, {}, layout=layout) if settings is None else settings
    # Nonexecuted CONFIG identity only, never an operation/producer placeholder.
    planned: Any = materialize_hpo_run(loaded, selected, learning_seed=config.learning_seeds[0],
        run_id='00000000-0000-0000-0000-000000000000').config
    return {'config_sha256': loaded.sha256,
        'effective_settings_sha256': selected.identity,
        'startup': _startup_geometry(planned.model.architecture(layout=layout), planned.model.inference,
            learner=planned.training.device, global_bytes=planned.run.global_cuda_budget_bytes),
        'operating': asdict(config.operating), 'host_admission': False, 'future_phase_checks_required': True}


def training_summary_data(state: Json, completion: Json, *, directory: Path,
                          completion_path: Path, completion_sha256: str,
                          clock_reference: Json) -> str:
    """Copy scalar cold DATA to an independent output string before owner close."""
    if (state['status'] != completion['state'] or state['updates'] != completion['committed_update']
            or state['generation_sha256'] != completion['generation_sha256']
            or state['status'] != 'complete' and completion['eligible']):
        raise ContractError('finalized state/completion summary differs')
    output = {'run_dir': str(directory), 'status': completion['state'], 'updates': completion['committed_update'],
        'accepted_decisions': state['accepted_decisions'], 'generation_sha256': completion['generation_sha256'],
        'elapsed_seconds': state['last_unix'] - state['created_unix'],
        'rust_player': {'completion_path': str(completion_path), 'completion_sha256': completion_sha256,
            'audit_path': completion['audit_path'], 'audit_sha256': completion['audit_sha256'],
            'runtime_binding_sha256': state['runtime_binding_sha256'], 'eligible': completion['eligible'], **clock_reference}}
    if 'used_decisions' in state:
        output.update(used_decisions=state['used_decisions'], collected_decisions=state['collected_decisions'],
            decision_budget=state['effective_config']['hpo_spec']['decision_budget'], stop_reason=state['stop_reason'])
    # No borrowed state/completion graph escapes into post-close printing.
    return canonical(output)


def _inputs(config: Any) -> Json:
    return {key: getattr(config.run, key) for key in
        ('admission_seconds', 'execution_seconds', 'wall_seconds', 'finalization_seconds')}


def _boot_id() -> str:
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _capture_clock() -> Json:
    from gf_common.rust_player import deadline_bits
    boot = _boot_id()
    before = time.monotonic()
    unix = time.time()
    after = time.monotonic()
    return {'boot_id': boot, 'monotonic_before_bits': deadline_bits(before),
        'monotonic_after_bits': deadline_bits(after), 'unix_anchor_bits': deadline_bits(unix)}


def _parent_mapping(ownership: Any) -> Json:
    from gf_training.lineage import _HELD_RUN_OWNERS, require_hpo_run_owner
    require_hpo_run_owner(ownership.directory, ownership)
    fd = _HELD_RUN_OWNERS[ownership.directory].descriptor
    actual = os.fstat(fd)
    return {'lock_kind': 'parent', 'fd': fd, 'device': actual.st_dev, 'inode': actual.st_ino}


def _snapshot(loaded: LoadedConfig, ownership: Any, *, fresh: bool) -> LoadedConfig:
    from gf_cli.config import load_rust_player_effective
    from gf_training.rust_player_supervision import write_control_record
    _parent_mapping(ownership)
    root = ownership.directory / '.rust-player'
    if root.is_symlink():
        raise ContractError('original ordinary control root must not be a symlink')
    if fresh:
        root.mkdir(mode=0o700)
        write_control_record(root / 'effective-config.json', loaded.effective)
    snapshot = load_rust_player_effective(root / 'effective-config.json', expected_sha256=loaded.sha256)
    if canonical(snapshot.effective) != canonical(loaded.effective):
        raise ContractError('original effective snapshot differs from current configuration')
    return snapshot


def _paused_run(ownership: Any, *, hpo: bool) -> Json:
    from gf_training.lineage import read_json
    from gf_training.rust_player_lineage import validate_run

    _parent_mapping(ownership)
    path = ownership.directory / 'run.json'
    if path.is_symlink():
        raise ContractError('ordinary paused state must be an original owned file')
    state = read_json(path)
    validate_run(state, hpo=hpo)
    checkpoint = state['effective_config']['hpo_spec']['operating'].get('checkpoint_evaluation') if hpo else None
    if state['status'] != 'paused' or not (state['updates'] or checkpoint and 0 in checkpoint['updates']):
        raise ContractError('actual paused run with a committed checkpoint required')
    return state


def _ordinary_resume_snapshot(invocation: LoadedConfig, ownership: Any, state: Json) -> LoadedConfig:
    from gf_cli.config import load_rust_player_effective

    path = ownership.directory / '.rust-player/effective-config.json'
    if path.parent.is_symlink() or path.is_symlink():
        raise ContractError('original ordinary control snapshot must not be a symlink')
    snapshot = load_rust_player_effective(path, expected_sha256=state['config_sha256'])
    requested = invocation.effective
    saved = snapshot.effective
    if 'ppo_update' not in requested['training']:
        saved = {**saved, 'training': {key: value for key, value in saved['training'].items()
            if key != 'ppo_update'}}
    if canonical(saved) != canonical(requested):
        raise ContractError('original effective snapshot differs from current configuration')
    return snapshot


def _resume_origin(loaded: LoadedConfig, ownership: Any, bound: Json, *,
                   hpo_context: Json | None = None, state: Json | None = None,
                   campaign_record_root: Path | None = None) -> tuple[Json, Json]:
    from gf_common.rust_player import deadline_seconds, hex_identity, validate_operation
    from gf_common.rust_player_clock import validate_operation_clock
    from gf_training.rust_player_supervision import read_control_record
    cfg: Any = loaded.config
    parent = ownership.directory
    hpo = hpo_context is not None
    if state is None:
        state = _paused_run(ownership, hpo=hpo)
    if (state['config_sha256'] != loaded.sha256
            or canonical(state['effective_config']) != canonical(loaded.effective)
            or canonical(state['config_inputs']) != canonical(loaded.inputs)
            or state['identity']['player_runtime'] != bound
            or state['runtime_binding_sha256'] != digest(bound)):
        raise ContractError('actual paused ordinary config/producer/snapshot differs')
    root = parent / '.rust-player'
    origin_path = root / 'operation-fresh.json'
    if campaign_record_root is not None:
        if not hpo:
            raise ContractError('campaign resume origin requires HPO')
        origins = [Path(entry['operation_path']) for entry in state['invocations'].values() if entry['mode'] == 'fresh']
        if (len(origins) != 1 or origins[0].parent != campaign_record_root
                or not origins[0].name.startswith('operation-')):
            raise ContractError('paused seed must have one original fresh operation in this supervisor session')
        root, origin_path = campaign_record_root, origins[0]
    referenced = [entry for entry in state['invocations'].values() if entry['operation_path'] == str(origin_path)]
    if len(referenced) != 1:
        raise ContractError('paused Run5 must reference its actual immutable fresh origin')
    origin, origin_sha = read_control_record(origin_path, root)
    validate_operation(origin)
    hex_identity(origin['operation_id'], 32)
    if (origin_sha != referenced[0]['operation_sha256'] or origin['schema_version'] != 2
            or origin['mode'] != ('hpo' if hpo else 'ordinary')
            or origin['operation_type'] != ('hpo-train' if hpo else 'ordinary-train')
            or origin['execution_request']['mode'] != 'fresh'
            or origin['execution_request'] != referenced[0]['execution_request']
            or origin['run_id'] != state['run_id'] or origin['parent_directory'] != str(parent)
            or origin['config_path'] != str(parent / '.rust-player/effective-config.json')
            or origin['config_sha256'] != loaded.sha256 or origin['runtime_binding_sha256'] != digest(bound)):
        raise ContractError('paused ordinary origin/config/root/operation hash differs')
    if hpo_context is not None and (origin['run_id'] != hpo_context['run_id']
            or origin['expected_context']['campaign_id'] != hpo_context['campaign_id']
            or origin['operation_id'] == hpo_context['operation_id']
            or any(entry['operation_path'] == str(root / f'operation-{hpo_context["operation_id"]}.json')
                for entry in state['invocations'].values())):
        raise ContractError('resume requires same experiment/run and distinct operation context')
    clock, _sha = read_control_record(root / f'clock-{origin["operation_id"]}.json', root)
    now = time.monotonic()
    if campaign_record_root is not None and now >= deadline_seconds(origin['deadlines']['admission_bits']):
        raise TimeoutError('original checkpoint training admission expired')
    validate_operation_clock(clock, operation=origin, session_clock_binding=clock['clock_binding'],
        session_finish=origin['deadlines']['finish_bits'], boot_id=_boot_id(), now=now)
    if (clock['inputs'] != _inputs(cfg) or clock['post_host_seconds'] != cfg.run.post_host_seconds
            or state['monotonic_authority'] != {'boot_id': clock['clock_binding']['boot_id'], 'deadlines': origin['deadlines']}
            or now >= deadline_seconds(origin['deadlines']['admission_bits'])):
        raise ContractError('original ordinary clock/allocation expired or differs')
    return origin, clock


@dataclass(frozen=True, eq=False)
class PreparedOrdinary:
    """Inspection DATA borrowing the actual live run_lock context, never JSON custody."""
    ownership: Any
    loaded: LoadedConfig
    operation: Json
    operation_path: Path
    clock: Json
    clock_path: Path
    clock_sha256: str
    provenance_path: Path
    runtime_binding: Json
    launch: Json
    companion: Any
    _launch_json: str | None = field(default=None, init=False, repr=False)
    _factory_companion: Any = field(default=None, init=False, repr=False)
    _companion_json: str | None = field(default=None, init=False, repr=False)

    def require(self) -> None:
        self._require_records([_parent_mapping(self.ownership)])

    def _require_records(self, locks: list[Json]) -> None:
        from gf_cli.config import load_rust_player_effective
        from gf_common.rust_player import deadline_seconds, validate_launch
        from gf_common.rust_player_clock import validate_operation_clock
        from gf_training.rust_player_supervision import read_control_record
        if self._launch_json is None or self._factory_companion is None:
            raise ContractError('actual ordinary factory preparation required')
        if (self.companion is not self._factory_companion
                or canonical(self.companion.manifest) != self._companion_json):
            raise ContractError('original discovered companion facts differ')
        if canonical(self.launch) != self._launch_json or self.launch['locks'] != locks:
            raise ContractError('launch differs from SAME original parent descriptor')
        validate_launch(self.launch)
        _config(self.loaded)
        root = self.ownership.directory / '.rust-player'
        op, checksum = read_control_record(self.operation_path, root)
        clock, clock_sha = read_control_record(self.clock_path, root)
        if (op != self.operation or checksum != self.launch['operation_sha256']
                or clock != self.clock or clock_sha != self.clock_sha256
                or op['parent_directory'] != str(self.ownership.directory)
                or op['config_path'] != str(root / 'effective-config.json')
                or op['runtime_binding_sha256'] != digest(self.runtime_binding)):
            raise ContractError('prepared immutable operation/clock/root differs')
        load_rust_player_effective(root / 'effective-config.json', expected_sha256=self.loaded.sha256)
        now = time.monotonic()
        validate_operation_clock(clock, operation=op, session_clock_binding=self.launch['clock_binding'],
            session_finish=self.launch['session_deadlines']['finish_bits'], boot_id=_boot_id(), now=now)
        if now >= deadline_seconds(op['deadlines']['admission_bits']):
            raise ContractError('original admission cutoff expired before exec')

    def execute(self) -> None:
        from gf_simulator_client.rust_player_companion import exec_supervisor
        self.require()
        exec_supervisor(self._factory_companion, self.launch)
        raise ContractError('fixed supervisor exec unexpectedly returned')

    def __reduce__(self) -> Any:
        raise TypeError('actual prepared run owner cannot be copied or serialized')


@contextmanager
def prepare_ordinary(loaded: LoadedConfig, directory: Path, *, mode: str,
                     pause_after_update: int | None = None) -> Iterator[PreparedOrdinary]:
    """Prepare under ONE real parent OFD; exec success cannot run context cleanup."""
    from gf_common.domain import integer
    from gf_common.rust_player import (
        BUILD_KEYS,
        runtime_binding,
        validate_launch,
        validate_operation,
    )
    from gf_common.rust_player_clock import clock_deadlines, operation_clock
    from gf_models.rust_player import sampling_runtime_identity
    from gf_simulator_client.rust_player_companion import discover_companion
    from gf_training.lineage import run_lock
    from gf_training.native_backend import binary_identity
    from gf_training.rust_player_supervision import (
        read_control_record,
        runtime_producer,
        write_control_record,
    )
    cfg, hpo, p = _config(loaded)
    if hpo or mode not in ('fresh', 'resume'):
        raise ContractError('explicit ordinary fresh/resume factory required')
    if pause_after_update is not None:
        integer(pause_after_update, 1)
    check_native_collection_storage(cfg.storage, Path(directory).absolute())
    startup_preview(loaded)
    native = binary_identity()
    companion = discover_companion(native)
    # Established actual build metadata only; no CUDA availability/init/tensors.
    producer = runtime_producer(companion, sampling_runtime_sha256=sampling_runtime_identity())
    bound = runtime_binding(cfg.run.player_runtime, producer)
    parent = Path(directory).resolve()
    op_id = uuid.uuid4().hex
    original_inputs = list(loaded.inputs)
    clock_binding = _capture_clock() if mode == 'fresh' else None
    fresh_deadlines = clock_deadlines(clock_binding, inputs=_inputs(cfg), post_host_seconds=p) if clock_binding is not None else None
    with run_lock(parent, create=mode == 'fresh') as ownership:
        root = parent / '.rust-player'
        if mode == 'resume':
            state = _paused_run(ownership, hpo=False)
            snapshot = _ordinary_resume_snapshot(loaded, ownership, state)
            origin, old_clock = _resume_origin(snapshot, ownership, bound, state=state)
            original_provenance, _ = read_control_record(root / f'inputs-{origin["operation_id"]}.json', root)
            if canonical(original_provenance) != canonical({'inputs': original_inputs}):
                raise ContractError('original ordinary input-file receipts differ')
            original_inputs = original_provenance['inputs']
            run_id, deadlines, clock_binding = origin['run_id'], origin['deadlines'], old_clock['clock_binding']
        else:
            snapshot = _snapshot(loaded, ownership, fresh=True)
            run_id, deadlines = str(uuid.uuid4()), fresh_deadlines
        assert clock_binding is not None and deadlines is not None
        operation = {'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': op_id,
            'run_id': run_id, 'mode': 'ordinary', 'operation_type': 'ordinary-train',
            'config_path': str(root / 'effective-config.json'), 'config_sha256': snapshot.sha256,
            'runtime_binding_sha256': digest(bound), 'deadlines': deadlines,
            'parent_directory': str(parent), 'expected_context': None,
            'execution_request': {'mode': mode, 'pause_after_update': pause_after_update}}
        validate_operation(operation)
        op_path = root / ('operation-fresh.json' if mode == 'fresh' else f'operation-{op_id}.json')
        op_sha = write_control_record(op_path, operation)
        clock = operation_clock(operation, clock_binding=clock_binding, inputs=_inputs(cfg), post_host_seconds=p)
        clock_path = root / f'clock-{op_id}.json'
        clock_sha = write_control_record(clock_path, clock)
        provenance = root / f'inputs-{op_id}.json'
        write_control_record(provenance, {'inputs': original_inputs})
        launch = {'schema_version': 1, 'kind': 'rust-player-launch-v1', 'supervision_id': uuid.uuid4().hex,
            'python_executable': str(Path(sys.executable).absolute()), 'operation_path': str(op_path),
            'operation_sha256': op_sha, 'record_directory': str(root), 'mode': 'ordinary',
            'session_deadlines': {'finish_bits': deadlines['finish_bits']}, 'clock_binding': clock_binding,
            'binding': {'python_abi': native['python_abi'], 'extension_sha256': native['extension_sha256'],
                'build': {key: native['build'][key] for key in BUILD_KEYS}}, 'locks': [_parent_mapping(ownership)]}
        validate_launch(launch)
        prepared = PreparedOrdinary(ownership, snapshot, operation, op_path, clock, clock_path,
            clock_sha, provenance, bound, launch, companion)
        # Only this actual prepare/discovery path issues fixed launch facts.
        # Constructors and dataclasses.replace retain DATA but cannot reissue it.
        object.__setattr__(prepared, '_launch_json', canonical(launch))
        object.__setattr__(prepared, '_factory_companion', companion)
        object.__setattr__(prepared, '_companion_json', canonical(companion.manifest))
        prepared.require()
        yield prepared


def exec_ordinary(loaded: LoadedConfig, directory: Path, *, mode: str,
                  pause_after_update: int | None = None) -> None:
    with prepare_ordinary(loaded, directory, mode=mode, pause_after_update=pause_after_update) as prepared:
        prepared.execute()


def _operation_namespace_data(loaded: LoadedConfig, context: Any, directory: Path) -> Json:
    """Immutable experiment-control DATA, never a campaign/proposal receipt."""
    from gf_common.rust_player import hex_identity, validate_context
    from gf_training.hpo_gpu_lease import JobContext
    if type(context) is not JobContext:
        raise ContractError('actual training JobContext DATA required')
    checked = validate_context(asdict(context))
    hex_identity(context.operation_id, 32)
    cfg: Any = loaded.config
    parent = Path(cfg.hpo_spec['storage']['parent_root']) / checked['run_id']
    if (parent != parent.absolute() or parent.name != cfg.run.name
            or str(parent.parent) != cfg.run.output_dir
            or directory == parent or directory.is_relative_to(parent)
            or parent.is_relative_to(directory)):
        raise ContractError('same configured HPO parent and separate stable control namespace required')
    return {'schema_version': 1, 'kind': 'hpo-operation-control-data-v1',
        'experiment_id': context.campaign_id, 'run_id': context.run_id,
        'parent_directory': str(parent), 'namespace_directory': str(directory), 'config_sha256': loaded.sha256,
        'config_inputs': list(loaded.inputs)}


def _operation_store_identity(store: Any) -> Json:
    from gf_cli.rust_player_campaign import _store_mapping
    mapping = _store_mapping(store)
    path = store.directory / 'lock'
    actual = path.lstat()
    if path.is_symlink() or (actual.st_dev, actual.st_ino) != (mapping['device'], mapping['inode']):
        raise ContractError('original operation Store lock pathname/descriptor differs')
    directory = store.directory.lstat()
    return {'directory_device': directory.st_dev, 'directory_inode': directory.st_ino,
        'lock_device': actual.st_dev, 'lock_inode': actual.st_ino}


@contextmanager
def _operation_namespace(loaded: LoadedConfig, context: Any, directory: Path, *,
                         mode: str) -> Iterator[Any]:
    from gf_cli.hpo_storage import CampaignStore
    directory = Path(directory).absolute()
    data = _operation_namespace_data(loaded, context, directory)
    cfg: Any = loaded.config
    with CampaignStore(directory, mode=mode,
            max_owned_bytes=cfg.hpo_spec['operating']['parent_budget_bytes']) as store:
        data['namespace_identity'] = _operation_store_identity(store)
        if mode == 'fresh':
            store.put('operation-control.json', data)
        elif store.get('operation-control.json')[0] != data:
            raise ContractError('original experiment namespace/config/inputs differ')
        yield store


@dataclass(frozen=True, eq=False)
class PreparedHpoOperation(PreparedOrdinary):
    """Borrowed inspection of the actual standalone HPO Store and parent owners."""
    store: Any = None
    _namespace_json: str | None = field(default=None, init=False, repr=False)

    def require(self) -> None:
        from gf_cli.rust_player_campaign import _store_mapping
        mapping = _store_mapping(self.store)
        data = self.store.get('operation-control.json')[0]
        if (self._namespace_json is None or canonical(data) != self._namespace_json
                or data['namespace_identity'] != _operation_store_identity(self.store)):
            raise ContractError('actual unchanged operation namespace required')
        self._require_records([mapping, _parent_mapping(self.ownership)])


def require_native_hpo_storage(storage: Any, identity: Json) -> None:
    from gf_cli.hpo_config import HPODiagnosticStorageConfig
    if isinstance(storage, HPODiagnosticStorageConfig):
        from gf_training.native_backend import CARD_INSTANCE_HPO_UPDATE_STORAGE, load_native

        native = load_native()
        if (identity['build'].get('card_instance_hpo_update_storage') != CARD_INSTANCE_HPO_UPDATE_STORAGE
                or not hasattr(getattr(native, 'StorageBundle', None), '_http_diagnostics')):
            raise ContractError('HPO diagnostics storage requires a matching native runtime wheel')


def require_native_hpo_action_profile(actions: object, build: Json) -> None:
    from gf_common.card_instance import (
        DREAM_COMPAT_PROFILE,
        MOON_DREAM_PROFILE,
        PESTLE_MIRAGE_PROFILE,
        ActionConfigV3,
    )
    from gf_training.native_backend import require_native_concealed_draw

    if not isinstance(actions, ActionConfigV3):
        return
    profile = actions.action_contract.profile_id
    if profile in (DREAM_COMPAT_PROFILE, MOON_DREAM_PROFILE, PESTLE_MIRAGE_PROFILE):
        require_native_concealed_draw(build)
    if profile == MOON_DREAM_PROFILE and build.get('card_instance_moon_dream') != MOON_DREAM_PROFILE:
        raise ContractError('Moon God and Dream profile requires a matching native runtime wheel')
    if profile == PESTLE_MIRAGE_PROFILE and build.get('card_instance_pestle_mirage') != PESTLE_MIRAGE_PROFILE:
        raise ContractError('Dangerous Pestle and Mirage profile requires a matching native runtime wheel')


@contextmanager
def prepare_hpo_operation(loaded: LoadedConfig, context: Any, namespace: Path, *, mode: str,
                          pause_after_update: int | None = None) -> Iterator[PreparedHpoOperation]:
    """Fixed hpo-train launch owning BOTH actual OFDs inside original clocks."""
    from gf_cli.hpo_config import sealed_storage
    from gf_cli.hpo_storage import check_storage
    from gf_cli.rust_player_campaign import _store_mapping
    from gf_common.domain import integer
    from gf_common.rust_player import (
        BUILD_KEYS,
        runtime_binding,
        validate_launch,
        validate_operation,
    )
    from gf_common.rust_player_clock import clock_deadlines, operation_clock
    from gf_models.rust_player import sampling_runtime_identity
    from gf_simulator_client.rust_player_companion import discover_companion
    from gf_training.lineage import run_lock
    from gf_training.native_backend import binary_identity
    from gf_training.rust_player_supervision import runtime_producer, write_control_record
    cfg, hpo, p = _config(loaded)
    if not hpo or cfg.model.model_id != 'mlp' or mode not in ('fresh', 'resume'):
        raise ContractError('explicit materialized HPO6 MLP fresh/resume factory required')
    if pause_after_update is not None:
        integer(pause_after_update, 1)
    namespace = Path(namespace).absolute()
    namespace_data = _operation_namespace_data(loaded, context, namespace)
    parent = Path(namespace_data['parent_directory'])
    check_native_collection_storage(cfg.storage, parent)
    startup_preview(loaded)
    storage = sealed_storage(cfg.hpo_spec['storage'])
    check_storage(storage, operation_dir=namespace, originals_root=Path(cfg.storage.db_root),
                  run_directory=parent, deadline=time.monotonic() + cfg.run.wall_seconds)
    native = binary_identity()
    require_native_hpo_action_profile(cfg.actions, native['build'])
    require_native_hpo_storage(storage, native)
    companion = discover_companion(native)
    producer = runtime_producer(companion, sampling_runtime_sha256=sampling_runtime_identity())
    bound = runtime_binding(cfg.hpo_spec['runtime'], producer)
    clock_binding = _capture_clock() if mode == 'fresh' else None
    fresh_deadlines = clock_deadlines(clock_binding, inputs=_inputs(cfg), post_host_seconds=p) if clock_binding is not None else None
    with _operation_namespace(loaded, context, namespace, mode=mode) as store, run_lock(parent, create=mode == 'fresh') as ownership:
        snapshot = _snapshot(loaded, ownership, fresh=mode == 'fresh')
        root = parent / '.rust-player'
        if mode == 'resume':
            origin, old_clock = _resume_origin(snapshot, ownership, bound, hpo_context=asdict(context))
            deadlines, clock_binding = origin['deadlines'], old_clock['clock_binding']
        else:
            deadlines = fresh_deadlines
        assert clock_binding is not None and deadlines is not None
        op_id = context.operation_id
        operation = {'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': op_id,
            'run_id': context.run_id, 'mode': 'hpo', 'operation_type': 'hpo-train',
            'config_path': str(root / 'effective-config.json'), 'config_sha256': snapshot.sha256,
            'runtime_binding_sha256': digest(bound), 'deadlines': deadlines,
            'parent_directory': str(parent), 'expected_context': asdict(context),
            'execution_request': {'mode': mode, 'pause_after_update': pause_after_update}}
        validate_operation(operation)
        op_path = root / ('operation-fresh.json' if mode == 'fresh' else f'operation-{op_id}.json')
        op_sha = write_control_record(op_path, operation)
        clock = operation_clock(operation, clock_binding=clock_binding, inputs=_inputs(cfg), post_host_seconds=p)
        clock_path = root / f'clock-{op_id}.json'
        clock_sha = write_control_record(clock_path, clock)
        provenance = root / f'inputs-{op_id}.json'
        write_control_record(provenance, {'inputs': list(loaded.inputs)})
        launch = {'schema_version': 1, 'kind': 'rust-player-launch-v1', 'supervision_id': uuid.uuid4().hex,
            'python_executable': str(Path(sys.executable).absolute()), 'operation_path': str(op_path),
            'operation_sha256': op_sha, 'record_directory': str(root), 'mode': 'hpo',
            'session_deadlines': {'finish_bits': deadlines['finish_bits']}, 'clock_binding': clock_binding,
            'binding': {'python_abi': native['python_abi'], 'extension_sha256': native['extension_sha256'],
                'build': {key: native['build'][key] for key in BUILD_KEYS}},
            'locks': [_store_mapping(store), _parent_mapping(ownership)]}
        validate_launch(launch)
        prepared = PreparedHpoOperation(ownership, snapshot, operation, op_path, clock, clock_path,
            clock_sha, provenance, bound, launch, companion, store=store)
        object.__setattr__(prepared, '_launch_json', canonical(launch))
        object.__setattr__(prepared, '_factory_companion', companion)
        object.__setattr__(prepared, '_companion_json', canonical(companion.manifest))
        object.__setattr__(prepared, '_namespace_json', canonical(store.get('operation-control.json')[0]))
        prepared.require()
        yield prepared
