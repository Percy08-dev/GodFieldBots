"""Sealed component declarations and typed files; no active-runtime authority.

Only recorded metadata is compared. No Torch, installed-version query, device
admission or current producer equality is used. Active C1 loaders remain strict.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from gf_common.action_contract import ActionContract
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    SIMULATOR_REVISION,
    ContractError,
    Json,
    canonical,
    closed,
    integer,
)
from gf_common.structured import GENERATOR_V2, INPUT_RECIPE
from gf_models.sharded_arrays import MAX_MANIFEST_BYTES, _pairs, validate_bundle_metadata
from gf_models.structured_binding import check_input_binding, input_binding
from gf_models.structured_config import (
    InferenceBindingV2,
    parameter_shapes,
    recorded_architecture_identity,
    recorded_model_config_v2,
    recorded_token_signature,
)

SEMANTIC_KEYS = {'input_recipe', 'generator_version', 'policy_input_version', 'end_projection_version', 'reward_policy_version',
                 'reviewed_spec_hashes', 'simulator_revision', 'catalog', 'token_signature', 'numeric_transform', 'runtime', 'numerics'}
CLASSIFICATION = {'action_contract', 'purpose', 'synthetic_capacity'}
NUMERICS = {'precision': 'ieee-fp32', 'tf32': False, 'amp': False, 'deterministic': True, 'cublas_workspace_config': ':4096:8', 'threads': 1}
ADAM = {'name': 'Adam', 'lr': .0003, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False,
        'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None, 'decoupled_weight_decay': False}


def checksum(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ContractError('sealed SHA256 required')
    return value


def basename(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}', value) is None or value in ('.', '..'):
        raise ContractError('sealed owned basename required')
    return value


def pruning_basename(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r'prune-[0-9a-f]{32}', value) is None:
        raise ContractError('owned unpredictable pruning staging basename required')
    return value


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class SealedScan:
    """One cumulative metadata + bounded shard/scratch admission and snapshot owner."""
    def __init__(self, *, max_owned_bytes: int, deadline: float) -> None:
        integer(max_owned_bytes, 1, 12 * 1024**3)
        self.limit, self.deadline, self.retained, self.peak = max_owned_bytes, deadline, 0, 0
        self.files: dict[Path, tuple[tuple[int, ...], str]] = {}
        self.inventories: dict[Path, tuple[str, ...]] = {}

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError('sealed inspection deadline')

    def charge(self, size: int, *, temporary: bool = False) -> None:
        self.check()
        integer(size, 0)
        if self.retained + size > self.limit:
            raise ContractError('sealed joint metadata/shard reservation exceeded')
        self.peak = max(self.peak, self.retained + size)
        if not temporary:
            self.retained += size

    @contextmanager
    def file(self, path: Path, expected: str | None = None) -> Iterator[tuple[BinaryIO, os.stat_result]]:
        self.check()
        if path.parent.is_symlink():
            raise ContractError('symlink sealed artifact directory')
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError('sealed regular file required')
            self.charge(min(before.st_size + 1, 1024**2), temporary=True)
            value = hashlib.sha256()
            while block := source.read(1024**2):
                self.check()
                value.update(block)
            sha = value.hexdigest()
            if expected is not None and sha != checksum(expected):
                raise ContractError('sealed file hash mismatch')
            prior = self.files.get(path)
            snapshot = (_identity(before), sha)
            if prior is not None and prior != snapshot:
                raise ContractError('sealed source changed since verified snapshot')
            if prior is None:
                self.charge(512 + len(str(path).encode()))
                self.files[path] = snapshot
            source.seek(0)
            yield source, before
            after, current = os.fstat(source.fileno()), path.lstat()
            if _identity(after) != _identity(before) or not stat.S_ISREG(current.st_mode) or _identity(current) != _identity(before):
                raise ContractError('sealed file/path changed while inspecting')

    def object(self, path: Path, expected: str | None = None) -> Json:
        with self.file(path, expected) as (source, info):
            if info.st_size > MAX_MANIFEST_BYTES:
                raise ContractError('sealed JSON exceeds64MiB')
            self.charge(32 * info.st_size + 65536)
            raw = source.read(info.st_size + 1)
            if len(raw) != info.st_size:
                raise ContractError('sealed JSON length changed')
            value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ContractError('nonfinite sealed JSON')))
            if not isinstance(value, dict):
                raise ContractError('sealed object required')
            return value

    def inventory(self, directory: Path, expected: set[str]) -> None:
        names, storage = self._inventory(directory, expected)
        previous = self.inventories.get(directory)
        if previous is not None and previous != names:
            raise ContractError('sealed inventory changed')
        if previous is None:
            self.charge(storage)
            self.inventories[directory] = names

    def _inventory(self, directory: Path, expected: set[str] | tuple[str, ...]) -> tuple[tuple[str, ...], int]:
        self.check()
        storage = 512 + 512 * len(expected)
        self.charge(storage, temporary=True)
        allowed, seen = set(expected), set()
        if directory.is_symlink():
            raise ContractError('sealed symlink directory')
        with os.scandir(directory) as entries:
            while True:
                self.check()
                self.charge(storage + 512, temporary=True)
                try:
                    entry = next(entries)
                except StopIteration:
                    break
                name = entry.name
                if (name not in allowed or name in seen or not entry.is_file(follow_symlinks=False)):
                    raise ContractError('sealed missing/foreign file inventory')
                storage += 512 + 4 * len(name)
                self.charge(storage, temporary=True)
                seen.add(name)
        if seen != allowed:
            raise ContractError('sealed missing/foreign file inventory')
        self.charge(storage + 512 * len(seen), temporary=True)
        names = tuple(sorted(seen))
        self.check()
        return names, storage

    def finalize(self) -> None:
        self.check()
        self.charge(128 * len(self.files) + 512, temporary=True)
        snapshots = list(self.files.items())
        for path, (_, expected) in snapshots:
            with self.file(path, expected):
                pass
        for directory, names in self.inventories.items():
            current, _ = self._inventory(directory, names)
            if current != names:
                raise ContractError('sealed final inventory changed')
        for path, (identity, _) in snapshots:
            self.check()
            if _identity(path.lstat()) != identity or path.is_symlink():
                raise ContractError('sealed post-sweep identity changed')
            self.check()
        self.check()

    def bundle(self, directory: Path, bundle: Json, *, optimizer_steps: int | None = None) -> Json:
        descriptors = validate_bundle_metadata(bundle, max_decoded_bytes=12 * 1024**3)
        self.charge(128 * len(descriptors) + 512)
        for name, sha in bundle['files'].items():
            basename(name)
            checksum(sha)
        for shard in bundle['shards']:
            name, metadata = shard['file'], shard['metadata']
            sidecar = self.object(directory / name.replace('.npz', '.json'), bundle['files'][name.replace('.npz', '.json')])
            if canonical(sidecar) != canonical(metadata):
                raise ContractError('sealed shard sidecar differs from inline metadata')
            with self.file(directory / name, bundle['files'][name]) as (source, info):
                decoded = sum(item['bytes'] for item in metadata['arrays'].values())
                self.charge(32 * info.st_size + 3 * decoded + 65536, temporary=True)
                with zipfile.ZipFile(source) as archive:
                    entries = archive.infolist()
                    if len(entries) != len(metadata['arrays']) or {item.filename for item in entries} != {key + '.npy' for key in metadata['arrays']}:
                        raise ContractError('sealed archive member ownership mismatch')
                    for entry in entries:
                        self.check()
                        spec = metadata['arrays'][entry.filename[:-4]]
                        if entry.file_size > spec['bytes'] + 4096:
                            raise ContractError('sealed NPY member exceeds admitted bytes')
                        with archive.open(entry) as raw:
                            version = np.lib.format.read_magic(raw)
                            reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                            if reader is None:
                                raise ContractError('unsupported sealed NPY version')
                            shape, _, dtype = reader(raw, max_header_size=4096)
                            if list(shape) != spec['shape'] or str(dtype) != spec['dtype'] or dtype.hasobject or raw.tell() + spec['bytes'] != entry.file_size:
                                raise ContractError('sealed NPY header/payload mismatch')
                    for entry in entries:
                        self.check()
                        encoded = archive.read(entry)
                        value = np.load(io.BytesIO(encoded), allow_pickle=False)
                        if not np.isfinite(value).all() or entry.filename.endswith('.exp_avg_sq.npy') and np.any(value < 0):
                            raise ContractError('invalid sealed tensor value/Adam variance')
                        if entry.filename.endswith('.step.npy') and optimizer_steps is not None:
                            integer(int(value), 1, optimizer_steps)
                        del value, encoded
                        self.check()
        return descriptors


def validate_recorded_semantics(value: Json, model: Any) -> None:
    """Validate supported static recipes and recorded runtime shape, no queries."""
    if not SEMANTIC_KEYS <= value.keys():
        raise ContractError('missing recorded semantic declaration')
    expected = {'input_recipe': INPUT_RECIPE, 'generator_version': GENERATOR_V2, 'policy_input_version': POLICY_INPUT_VERSION,
        'end_projection_version': END_PROJECTION_VERSION, 'reward_policy_version': 'natural-terminal-v1',
        'reviewed_spec_hashes': REVIEWED_SPEC_HASHES, 'simulator_revision': SIMULATOR_REVISION, 'catalog': CATALOG_META,
        'token_signature': recorded_token_signature(model), 'numeric_transform': 'signed-log1p-fp32-v1', 'numerics': NUMERICS}
    if any(canonical(value[key]) != canonical(item) for key, item in expected.items()):
        raise ContractError('unsupported sealed semantic declaration')
    runtime = closed(value['runtime'], {'torch', 'numpy', 'python', 'gil_enabled'})
    if any(type(runtime[key]) is not str or not runtime[key] for key in ('torch', 'numpy', 'python')) or type(runtime['gil_enabled']) is not bool:
        raise ContractError('invalid recorded runtime shape')


def _semantics(value: Json, model: Any) -> None:
    validate_recorded_semantics(value, model)
    producer = closed(value['producer'], {'code_revision', 'source_digest'})
    checksum(producer['source_digest'])
    if producer['code_revision'] is not None and (type(producer['code_revision']) is not str or re.fullmatch(r'[0-9a-f]{40}', producer['code_revision']) is None):
        raise ContractError('invalid recorded producer')


@dataclass(frozen=True)
class SealedArtifact:
    inference: Json
    resume: Json | None
    files: Json
    retained: bool


def _inference_metadata(directory: Path, expected_inference_digest: str, action_contract: ActionContract,
                        purpose: str, synthetic_capacity: bool, scan: SealedScan) -> tuple[Json, Any, Json]:
    checksum(expected_inference_digest)
    input_binding(action_contract, purpose, synthetic_capacity)
    inference = scan.object(directory / 'inference.json', expected_inference_digest)
    closed(inference, {'schema_version', 'kind', 'model', 'architecture_identity', 'action_config_hash', 'producer', 'weights'} | SEMANTIC_KEYS | CLASSIFICATION)
    check_input_binding({key: inference[key] for key in CLASSIFICATION}, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    if type(inference['schema_version']) is not int or inference['schema_version'] != 3 or inference['kind'] != 'structured-inference':
        raise ContractError('sealed inference3 required')
    model = recorded_model_config_v2(inference['model'], inference['token_signature'])
    _semantics(inference, model)
    if inference['architecture_identity'] != recorded_architecture_identity(model):
        raise ContractError('sealed architecture identity mismatch')
    checksum(inference['action_config_hash'])
    if inference['weights'].get('prefix') != 'weights':
        raise ContractError('sealed inference weights prefix mismatch')
    weights = validate_bundle_metadata(inference['weights'])
    expected_weights = parameter_shapes(model)
    if set(weights) != set(expected_weights) or any(weights[key]['dtype'] != 'float32' or weights[key]['shape'] != list(shape) for key, shape in expected_weights.items()):
        raise ContractError('sealed named weight descriptors mismatch')
    return inference, model, expected_weights


def inspect_retained_artifact(directory: Path, *, expected_inference_digest: str, action_contract: ActionContract,
                              purpose: str, synthetic_capacity: bool, scan: SealedScan) -> SealedArtifact:
    inference, model, expected_weights = _inference_metadata(directory, expected_inference_digest, action_contract,
                                                            purpose, synthetic_capacity, scan)
    resume = scan.object(directory / 'resume.json')
    closed(resume, {'schema_version', 'kind', 'inference_digest', 'model', 'binding', 'resume_identity', 'producer',
        'optimizer_steps', 'optimizer_recipe', 'parameter_states', 'cuda_rng_count', 'shuffle_rng', 'state'} | SEMANTIC_KEYS | CLASSIFICATION)
    check_input_binding({key: resume[key] for key in CLASSIFICATION}, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    checksum(resume['inference_digest'])
    if (type(resume['schema_version']) is not int or resume['schema_version'] != 3 or resume['kind'] != 'structured-resume'
            or resume['inference_digest'] != expected_inference_digest or canonical(resume['model']) != canonical(inference['model'])
            or any(canonical(resume[key]) != canonical(inference[key]) for key in SEMANTIC_KEYS)
            or canonical(resume['optimizer_recipe']) != canonical(ADAM)):
        raise ContractError('sealed paired resume/inference declarations mismatch')
    _semantics(resume, model)
    if (resume['producer']['source_digest'] != inference['producer']['source_digest']
            or resume['producer']['code_revision'] is not None and inference['producer']['code_revision'] is not None
            and resume['producer']['code_revision'] != inference['producer']['code_revision']):
        raise ContractError('sealed paired producer declarations mismatch')
    binding = InferenceBindingV2(**closed(resume['binding'], set(InferenceBindingV2.__dataclass_fields__)))
    checksum(resume['resume_identity'])
    integer(resume['optimizer_steps'], 0, 2**31 - 1)
    integer(resume['cuda_rng_count'], 0, 1024)
    if (binding.device == 'cpu' and resume['cuda_rng_count'] != 0
            or binding.device != 'cpu' and resume['cuda_rng_count'] == 0 or resume['state'].get('prefix') != 'resume'):
        raise ContractError('sealed recorded device/RNG/state prefix mismatch')
    states = resume['parameter_states']
    if not isinstance(states, dict) or set(states) != set(expected_weights):
        raise ContractError('sealed Adam parameter ownership mismatch')
    expected: Json = {'rng.torch_cpu': {'dtype': 'uint8', 'shape': None}}
    expected.update({f'rng.torch_cuda.{index}': {'dtype': 'uint8', 'shape': None} for index in range(resume['cuda_rng_count'])})
    for name, shape in expected_weights.items():
        state = closed(states[name], {'present'})
        if type(state['present']) is not bool:
            raise ContractError('sealed Adam presence boolean required')
        if state['present']:
            expected['adam.' + name + '.step'] = {'dtype': 'int64', 'shape': []}
            for key in ('exp_avg', 'exp_avg_sq'):
                expected[f'adam.{name}.{key}'] = {'dtype': 'float32', 'shape': list(shape)}
    descriptors = validate_bundle_metadata(resume['state'])
    if set(expected) != set(descriptors) or any(descriptors[key]['dtype'] != item['dtype'] or item['shape'] is not None and descriptors[key]['shape'] != item['shape'] for key, item in expected.items()):
        raise ContractError('sealed Adam/RNG descriptors mismatch')
    for key in expected:
        if key.startswith('rng.') and (len(descriptors[key]['shape']) != 1 or descriptors[key]['shape'][0] == 0):
            raise ContractError('sealed RNG byte vector required')
    shuffle = closed(resume['shuffle_rng'], {'bit_generator', 'state', 'has_uint32', 'uinteger'})
    nested = closed(shuffle['state'], {'state', 'inc'})
    if shuffle['bit_generator'] != 'PCG64':
        raise ContractError('sealed PCG64 required')
    integer(nested['state'], 0, 2**128 - 1)
    integer(nested['inc'], 1, 2**128 - 1)
    if nested['inc'] % 2 != 1:
        raise ContractError('sealed PCG64 odd increment required')
    integer(shuffle['has_uint32'], 0, 1)
    integer(shuffle['uinteger'], 0, 2**32 - 1)
    files = {**inference['weights']['files'], **resume['state']['files'], 'inference.json': expected_inference_digest,
             'resume.json': scan.files[directory / 'resume.json'][1]}
    scan.inventory(directory, set(files))
    scan.bundle(directory, inference['weights'])
    scan.bundle(directory, resume['state'], optimizer_steps=resume['optimizer_steps'])
    scan.finalize()
    return SealedArtifact(inference, resume, files, True)


def inspect_pruned_inference(directory: Path, *, expected_inference_digest: str, expected_resume_digest: str,
                             original_files: Json, deletion_receipt: Json, action_contract: ActionContract,
                             purpose: str, synthetic_capacity: bool, scan: SealedScan) -> SealedArtifact:
    checksum(expected_inference_digest)
    checksum(expected_resume_digest)
    inference, _, _ = _inference_metadata(directory, expected_inference_digest, action_contract, purpose, synthetic_capacity, scan)
    if not isinstance(original_files, dict):
        raise ContractError('original sealed file map required')
    scan.charge(512 + len(original_files) * 256)
    for name, sha in original_files.items():
        basename(name)
        checksum(sha)
    required = {'inference.json': expected_inference_digest, 'resume.json': expected_resume_digest, **inference['weights']['files']}
    if not set(required).issubset(original_files) or any(original_files[name] != sha for name, sha in required.items()):
        raise ContractError('pruned original inference/resume file map mismatch')
    states = set(original_files) - set(required)
    count = len(states) // 2
    if not count or states != {f'resume-{index:06}.{suffix}' for index in range(count) for suffix in ('npz', 'json')}:
        raise ContractError('pruned original resume shard map invalid')
    receipt = closed(deletion_receipt, {'status', 'inference_sha256', 'original_files', 'deletion_intent', 'deletion_receipt'})
    intent = closed(receipt['deletion_intent'], {'files', 'staging_basename'})
    staging = pruning_basename(intent['staging_basename'])
    deleted = {name: sha for name, sha in original_files.items() if name != 'inference.json'}
    if (receipt['status'] != 'deleted' or receipt['inference_sha256'] != expected_inference_digest
            or canonical(receipt['original_files']) != canonical(original_files)
            or canonical(intent['files']) != canonical(deleted)
            or canonical(receipt['deletion_receipt']) != canonical({'files': deleted})):
        raise ContractError('pruned completed deletion receipt required')
    try:
        (directory.parent / staging).lstat()
    except FileNotFoundError:
        pass
    else:
        raise ContractError('completed pruning staging entry must be absent')
    scan.inventory(directory, {'inference.json'})
    scan.finalize()
    return SealedArtifact(inference, None, dict(original_files), False)
