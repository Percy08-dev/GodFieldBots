"""Internal public-launch composition; public execution routing stays guarded.

Configuration previews are DATA, not admission or a physical-capacity promise.
Torch/native/process imports belong only to the explicit ordinary preparation.
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


def startup_preview(loaded: LoadedConfig) -> Json:
    """Known fresh network demand only; no Torch, hardware or live account."""
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.rust_player_device import CudaCheckpointProfile

    cfg, hpo, _p = _config(loaded)
    profile = CudaCheckpointProfile(cfg.model.architecture(layout=layout), cfg.model.inference, layout=layout)
    learner, serving = cfg.training.device, cfg.model.inference.device
    active = learner != 'cpu' or serving != 'cpu'
    global_bytes = (cfg.hpo_spec['operating']['managed_cuda_bytes'] if hpo
        else cfg.run.global_cuda_budget_bytes) if active else 0
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
        'declared_ram_bytes': cfg.hpo_spec['operating']['managed_ram_bytes'] if hpo else 24 * 1024**3,
        'declared_source_bytes': cfg.preprocessing.memory_budget_mib * 1024**2,
        'declared_storage_bytes': cfg.storage.settings().budget_bytes,
        'host_admission': False, 'future_phase_checks_required': True}


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


def _resume_origin(loaded: LoadedConfig, ownership: Any, bound: Json) -> tuple[Json, Json]:
    from gf_common.rust_player import deadline_seconds, hex_identity, validate_operation
    from gf_common.rust_player_clock import validate_operation_clock
    from gf_training.lineage import read_json
    from gf_training.rust_player_lineage import validate_run
    from gf_training.rust_player_supervision import read_control_record
    cfg: Any = loaded.config
    parent = ownership.directory
    path = parent / 'run.json'
    if path.is_symlink():
        raise ContractError('ordinary paused state must be an original owned file')
    state = read_json(path)
    validate_run(state)
    if (state['status'] != 'paused' or not state['updates']
            or state['config_sha256'] != loaded.sha256
            or canonical(state['effective_config']) != canonical(loaded.effective)
            or canonical(state['config_inputs']) != canonical(loaded.inputs)
            or state['identity']['player_runtime'] != bound
            or state['runtime_binding_sha256'] != digest(bound)):
        raise ContractError('actual paused ordinary config/producer/snapshot differs')
    root = parent / '.rust-player'
    origin_path = root / 'operation-fresh.json'
    referenced = [entry for entry in state['invocations'].values() if entry['operation_path'] == str(origin_path)]
    if len(referenced) != 1:
        raise ContractError('paused Run5 must reference its actual immutable fresh origin')
    origin, origin_sha = read_control_record(origin_path, root)
    validate_operation(origin)
    hex_identity(origin['operation_id'], 32)
    if (origin_sha != referenced[0]['operation_sha256'] or origin['schema_version'] != 2
            or origin['mode'] != 'ordinary' or origin['operation_type'] != 'ordinary-train'
            or origin['execution_request']['mode'] != 'fresh'
            or origin['execution_request'] != referenced[0]['execution_request']
            or origin['run_id'] != state['run_id'] or origin['parent_directory'] != str(parent)
            or origin['config_path'] != str(root / 'effective-config.json')
            or origin['config_sha256'] != loaded.sha256 or origin['runtime_binding_sha256'] != digest(bound)):
        raise ContractError('paused ordinary origin/config/root/operation hash differs')
    clock, _sha = read_control_record(root / f'clock-{origin["operation_id"]}.json', root)
    now = time.monotonic()
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
        from gf_cli.config import load_rust_player_effective
        from gf_common.rust_player import deadline_seconds, validate_launch
        from gf_common.rust_player_clock import validate_operation_clock
        from gf_training.rust_player_supervision import read_control_record
        if self._launch_json is None or self._factory_companion is None:
            raise ContractError('actual ordinary factory preparation required')
        if (self.companion is not self._factory_companion
                or canonical(self.companion.manifest) != self._companion_json):
            raise ContractError('original discovered companion facts differ')
        mapping = _parent_mapping(self.ownership)
        if canonical(self.launch) != self._launch_json or self.launch['locks'] != [mapping]:
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
    from gf_training.rust_player_supervision import runtime_producer, write_control_record
    cfg, hpo, p = _config(loaded)
    if hpo or mode not in ('fresh', 'resume'):
        raise ContractError('explicit ordinary fresh/resume factory required')
    if pause_after_update is not None:
        integer(pause_after_update, 1)
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
        snapshot = _snapshot(loaded, ownership, fresh=mode == 'fresh')
        root = parent / '.rust-player'
        if mode == 'resume':
            origin, old_clock = _resume_origin(snapshot, ownership, bound)
            run_id, deadlines, clock_binding = origin['run_id'], origin['deadlines'], old_clock['clock_binding']
        else:
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
