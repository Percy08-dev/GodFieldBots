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
from gf_common.card_instance import ActionConfigV3, CardInstanceContract
from gf_common.card_instance_rules import CardInstanceRules
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
from gf_common.structured_layout import StructuredLayout, layout_for_contract
from gf_models.sharded_arrays import MAX_MANIFEST_BYTES, _pairs, validate_bundle_metadata
from gf_models.structured_binding import (
    check_input_binding,
    check_input_binding_game_v3,
    check_input_binding_v3,
    input_binding,
    input_binding_game_v3,
    input_binding_v3,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    model_config_v3,
    parameter_shapes,
    recorded_architecture_identity,
    recorded_model_config_v2,
    recorded_token_signature,
)

SEMANTIC_KEYS = {'input_recipe', 'generator_version', 'policy_input_version', 'end_projection_version', 'reward_policy_version',
                 'reviewed_spec_hashes', 'simulator_revision', 'catalog', 'token_signature', 'numeric_transform', 'runtime', 'numerics'}
CLASSIFICATION = {'action_contract', 'purpose', 'synthetic_capacity'}
SEMANTIC_KEYS_V3 = SEMANTIC_KEYS | {'rule_signature', 'spec_inventory_sha256'}
CLASSIFICATION_V3 = CLASSIFICATION | {'rule_signature'}
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
        self.directory_inventories: dict[Path, tuple[Any,tuple[str,...]]] = {}

    def verified_digest(self, path: Path) -> str:
        """Digest from an explicitly admitted physical original."""
        self.check()
        if path not in self.files:
            raise ContractError('unregistered verified original digest')
        return self.files[path][1]

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

    def directories(self, path: Path, expected: set[str]) -> None:
        self.check()
        self.charge(1024+1024*len(expected))
        before=path.lstat()
        if not stat.S_ISDIR(before.st_mode) or path.is_symlink():raise ContractError('sealed directory namespace required')
        seen=set()
        with os.scandir(path) as entries:
            for entry in entries:
                self.check()
                if entry.name not in expected or entry.name in seen or not entry.is_dir(follow_symlinks=False):
                    raise ContractError('sealed directory inventory differs')
                seen.add(entry.name)
        names=tuple(sorted(seen))
        if seen!=expected:raise ContractError('sealed directory inventory differs')
        if _identity(path.lstat())!=_identity(before):raise ContractError('sealed directory changed')
        prior=self.directory_inventories.get(path)
        value=(_identity(before),names)
        if prior is not None and prior!=value:raise ContractError('sealed directory inventory changed')
        self.directory_inventories[path]=value

    def recheck_directories(self) -> None:
        for path,(identity,names) in self.directory_inventories.items():
            self.check()
            if path.is_symlink() or _identity(path.lstat())!=identity or tuple(sorted(entry.name for entry in path.iterdir()))!=names:
                raise ContractError('sealed directory namespace changed')

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
        self.recheck_directories()
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


def static_semantics_v3(model: Any, *, layout: StructuredLayout, action_contract: CardInstanceContract,
                        rules: CardInstanceRules) -> Json:
    from dataclasses import asdict

    from gf_common.card_instance import CATALOG_META as CARD_CATALOG
    from gf_common.card_instance_rules import rule_signature
    model_config_v3(asdict(model), layout=layout)
    if layout is not layout_for_contract(action_contract):
        raise ContractError("recorded model3 profile/layout mismatch")
    from gf_common.card_instance import reviewed_specs
    specs = reviewed_specs(action_contract)
    return {"input_recipe": layout.input_recipe, "generator_version": layout.generator,
        "policy_input_version": layout.policy_input_version, "end_projection_version": END_PROJECTION_VERSION,
        "reward_policy_version": "natural-terminal-v1", "reviewed_spec_hashes": specs,
        "simulator_revision": action_contract.simulator_revision, "catalog": CARD_CATALOG,
        "token_signature": recorded_token_signature(model, layout=layout), "numeric_transform": "signed-log1p-fp32-v1",
        "numerics": NUMERICS, "rule_signature": rule_signature(action_contract, rules),
        "spec_inventory_sha256": action_contract.spec_inventory_sha256}


def static_semantics_v4(model: Any, *, layout: StructuredLayout, action_contract: CardInstanceContract,
                        rules: CardInstanceRules) -> Json:
    """Only inference semantics; reward and optimizer are resume5 training fields."""
    from gf_models.hpo_config import _legacy_validation_proxy, validate_model_v4
    validate_model_v4(model, layout=layout)
    result = static_semantics_v3(_legacy_validation_proxy(model), layout=layout,
        action_contract=action_contract, rules=rules)
    result.pop("reward_policy_version")
    return result


@dataclass(frozen=True)
class _SealedContextV3:
    layout: StructuredLayout
    action_contract: CardInstanceContract
    rules: CardInstanceRules
    purpose: str
    synthetic_capacity: bool
    action_config_hash: str
    game: bool = False

    @property
    def schema(self) -> int:
        return 4

    @property
    def semantic_keys(self) -> set[str]:
        return SEMANTIC_KEYS_V3

    def model(self, raw: Json) -> Any:
        return model_config_v3(raw, layout=self.layout)

    def semantics(self, model: Any) -> Json:
        return static_semantics_v3(model, layout=self.layout, action_contract=self.action_contract, rules=self.rules)

    def resume_metadata(self) -> Json:
        return {}

    def optimizer(self, model: Any) -> Json:
        return ADAM

    @property
    def class_keys(self) -> set[str]:
        return CLASSIFICATION_V3 | ({"layout_signature"} if self.game else set())

    def check(self, raw: Json) -> None:
        checksum(self.action_config_hash)
        if self.game:
            check_input_binding_game_v3(raw, action_contract=self.action_contract, rules=self.rules)
        else:
            check_input_binding_v3(raw, action_contract=self.action_contract, rules=self.rules,
                purpose=self.purpose, synthetic_capacity=self.synthetic_capacity)


@dataclass(frozen=True)
class _SealedContextV4(_SealedContextV3):
    training_recipe: Json | None = None
    optimizer_recipe: Json | None = None

    @property
    def schema(self) -> int:
        return 5

    @property
    def semantic_keys(self) -> set[str]:
        return SEMANTIC_KEYS_V3 - {"reward_policy_version"}

    def model(self, raw: Json) -> Any:
        from gf_models.structured_binding import recorded_inference_model_v4
        return recorded_inference_model_v4(raw, layout=self.layout)

    def semantics(self, model: Any) -> Json:
        return static_semantics_v4(model, layout=self.layout, action_contract=self.action_contract, rules=self.rules)

    def resume_metadata(self) -> Json:
        if type(self.training_recipe) is not dict:
            raise ContractError("sealed resume5 requires an expected closed training recipe")
        return {"training_recipe": self.training_recipe}

    def optimizer(self, model: Any) -> Json:
        from gf_models.structured_artifacts import _optimizer_descriptor
        if self.optimizer_recipe is None:
            raise ContractError("sealed resume5 requires the expected optimizer recipe")
        return _optimizer_descriptor(model, self.optimizer_recipe)


def validate_recorded_semantics_v3(value: Json, model: Any, *, layout: StructuredLayout,
                                   action_contract: CardInstanceContract, rules: CardInstanceRules) -> None:
    if not SEMANTIC_KEYS_V3 <= value.keys():
        raise ContractError("missing recorded format4 semantic declaration")
    expected = static_semantics_v3(model, layout=layout, action_contract=action_contract, rules=rules)
    if any(canonical(value[key]) != canonical(item) for key, item in expected.items()):
        raise ContractError("unsupported recorded format4 semantics")
    runtime = closed(value["runtime"], {"torch", "numpy", "python", "gil_enabled"})
    if any(type(runtime[key]) is not str or not runtime[key] for key in ("torch", "numpy", "python")) or type(runtime["gil_enabled"]) is not bool:
        raise ContractError("invalid recorded runtime shape")


def _semantics(value: Json, model: Any, context: _SealedContextV3 | None = None) -> None:
    if context is None:
        validate_recorded_semantics(value, model)
    else:
        if not context.semantic_keys <= value.keys() or any(canonical(value[key]) != canonical(item)
                for key, item in context.semantics(model).items()):
            raise ContractError("unsupported recorded structured semantics")
        runtime = closed(value["runtime"], {"torch", "numpy", "python", "gil_enabled"})
        if any(type(runtime[key]) is not str or not runtime[key] for key in ("torch", "numpy", "python")) or type(runtime["gil_enabled"]) is not bool:
            raise ContractError("invalid recorded runtime shape")
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


def _inference_metadata(directory: Path, expected_inference_digest: str, action_contract: Any,
                        purpose: str, synthetic_capacity: bool, scan: SealedScan, context: _SealedContextV3 | None = None) -> tuple[Json, Any, Json]:
    checksum(expected_inference_digest)
    if context is None:
        input_binding(action_contract, purpose, synthetic_capacity)
    elif context.game:
        input_binding_game_v3(context.action_contract, context.rules)
    else:
        input_binding_v3(context.action_contract, context.rules, purpose, synthetic_capacity)
    inference = scan.object(directory / 'inference.json', expected_inference_digest)
    return _inference_document(inference, action_contract, purpose, synthetic_capacity, context)


def _inference_document(inference: Json, action_contract: Any, purpose: str, synthetic_capacity: bool,
                        context: _SealedContextV3 | None = None) -> tuple[Json, Any, Json]:
    semantic_keys, class_keys = (SEMANTIC_KEYS, CLASSIFICATION) if context is None else (context.semantic_keys, context.class_keys)
    schema = 3 if context is None else context.schema
    closed(inference, {'schema_version', 'kind', 'model', 'architecture_identity', 'action_config_hash', 'producer', 'weights'} | semantic_keys | class_keys)
    if context is None:
        check_input_binding({key: inference[key] for key in class_keys}, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    else:
        context.check({key: inference[key] for key in class_keys})
        if inference["action_config_hash"] != context.action_config_hash:
            raise ContractError("recorded format4 action Config mismatch")
    if type(inference['schema_version']) is not int or inference['schema_version'] != schema or inference['kind'] != 'structured-inference':
        raise ContractError('sealed inference3 required')
    model = (recorded_model_config_v2(inference['model'], inference['token_signature']) if context is None else
             context.model(inference['model']))
    if context is not None and context.game and model.max_positions > 18:
        raise ContractError("game sealed model capacity exceeds real18")
    options: dict[str, Any] = {} if context is None else {"layout": context.layout}
    _semantics(inference, model, context)
    if inference['architecture_identity'] != recorded_architecture_identity(model, **options):
        raise ContractError('sealed architecture identity mismatch')
    checksum(inference['action_config_hash'])
    if inference['weights'].get('prefix') != 'weights':
        raise ContractError('sealed inference weights prefix mismatch')
    weights = validate_bundle_metadata(inference['weights'])
    expected_weights = parameter_shapes(model, **options)
    if set(weights) != set(expected_weights) or any(weights[key]['dtype'] != 'float32' or weights[key]['shape'] != list(shape) for key, shape in expected_weights.items()):
        raise ContractError('sealed named weight descriptors mismatch')
    return inference, model, expected_weights


def _inspect_retained_artifact(directory: Path, *, expected_inference_digest: str, action_contract: Any,
                              purpose: str, synthetic_capacity: bool, scan: SealedScan, context: _SealedContextV3 | None = None) -> SealedArtifact:
    semantic_keys, class_keys = (SEMANTIC_KEYS, CLASSIFICATION) if context is None else (context.semantic_keys, context.class_keys)
    schema = 3 if context is None else context.schema
    inference, model, expected_weights = _inference_metadata(directory, expected_inference_digest, action_contract,
                                                            purpose, synthetic_capacity, scan, context)
    resume = scan.object(directory / 'resume.json')
    metadata = {} if context is None else context.resume_metadata()
    optimizer_recipe = ADAM if context is None else context.optimizer(model)
    closed(resume, {'schema_version', 'kind', 'inference_digest', 'model', 'binding', 'resume_identity', 'producer',
        'optimizer_steps', 'optimizer_recipe', 'parameter_states', 'cuda_rng_count', 'shuffle_rng', 'state'} | ({'action_config_hash'} if context is not None else set()) | semantic_keys | class_keys | set(metadata))
    if context is None:
        check_input_binding({key: resume[key] for key in class_keys}, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    else:
        context.check({key: resume[key] for key in class_keys})
        if resume["action_config_hash"] != context.action_config_hash:
            raise ContractError("recorded resume4 action Config mismatch")
    checksum(resume['inference_digest'])
    paired_model = resume["model"]
    if schema == 5:
        from gf_models.hpo_config import model_config_v4
        from gf_models.structured_binding import inference_model
        assert context is not None
        full_model = model_config_v4(paired_model, layout=context.layout)
        paired_model = inference_model(full_model)
    if (type(resume['schema_version']) is not int or resume['schema_version'] != schema or resume['kind'] != 'structured-resume'
            or resume['inference_digest'] != expected_inference_digest or canonical(paired_model) != canonical(inference['model'])
            or any(canonical(resume[key]) != canonical(inference[key]) for key in semantic_keys)
            or canonical(resume['optimizer_recipe']) != canonical(optimizer_recipe)
            or any(canonical(resume[key]) != canonical(value) for key, value in metadata.items())):
        raise ContractError('sealed paired resume/inference declarations mismatch')
    _semantics(resume, model, context)
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
                             original_files: Json, deletion_receipt: Json, action_contract: Any,
                             purpose: str, synthetic_capacity: bool, scan: SealedScan,
                             _context: _SealedContextV3 | None = None) -> SealedArtifact:
    checksum(expected_inference_digest)
    checksum(expected_resume_digest)
    inference, _, _ = _inference_metadata(directory, expected_inference_digest, action_contract, purpose, synthetic_capacity, scan, _context)
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



def inspect_retained_artifact(directory: Path, *, expected_inference_digest: str, action_contract: ActionContract,
                              purpose: str, synthetic_capacity: bool, scan: SealedScan) -> SealedArtifact:
    return _inspect_retained_artifact(directory, expected_inference_digest=expected_inference_digest,
        action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity, scan=scan)


def inspect_retained_artifact_v3(directory: Path, *, expected_inference_digest: str,
                                action_contract: CardInstanceContract, rules: CardInstanceRules,
                                purpose: str, synthetic_capacity: bool, action_config_hash: str,
                                layout: StructuredLayout, scan: SealedScan) -> SealedArtifact:
    context = _SealedContextV3(layout, action_contract, rules, purpose, synthetic_capacity, action_config_hash)
    input_binding_v3(action_contract, rules, purpose, synthetic_capacity)
    if layout is not layout_for_contract(action_contract):
        raise ContractError("recorded format4 profile/layout mismatch")
    return _inspect_retained_artifact(directory, expected_inference_digest=expected_inference_digest,
        action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity, scan=scan, context=context)


def validate_game_inference_metadata(inference: Json, *, actions: ActionConfigV3,
                                      rules: CardInstanceRules) -> Any:
    """Recorded-only format4 game declaration, without files/Torch/runtime queries."""
    from dataclasses import asdict

    from gf_common.domain import digest
    context = _SealedContextV3(layout_for_contract(actions.action_contract), actions.action_contract,
        rules, "qualified_game", False, digest(asdict(actions)), True)
    input_binding_game_v3(actions.action_contract, rules)
    if actions.max_sequence_length > 18:
        raise ContractError("game sealed actions exceed real18")
    _, model, _ = _inference_document(inference, actions.action_contract, "qualified_game", False, context)
    return model


def inspect_retained_game_inference_v3(directory: Path, *, expected_inference_digest: str,
                                        actions: ActionConfigV3, rules: CardInstanceRules,
                                        scan: SealedScan) -> SealedArtifact:
    """Own just inference and its referenced shards; a resume is not required."""
    inference = scan.object(directory / "inference.json", expected_inference_digest)
    validate_game_inference_metadata(inference, actions=actions, rules=rules)
    scan.bundle(directory, inference["weights"])
    files = {"inference.json": expected_inference_digest, **inference["weights"]["files"]}
    scan.finalize()
    return SealedArtifact(inference, None, files, True)


def inspect_retained_game_artifact_v3(directory: Path, *, expected_inference_digest: str,
                                      actions: ActionConfigV3, rules: CardInstanceRules,
                                      scan: SealedScan) -> SealedArtifact:
    from dataclasses import asdict

    from gf_common.domain import digest
    input_binding_game_v3(actions.action_contract, rules)
    context = _SealedContextV3(layout_for_contract(actions.action_contract), actions.action_contract,
        rules, 'qualified_game', False, digest(asdict(actions)), True)
    return _inspect_retained_artifact(directory, expected_inference_digest=expected_inference_digest,
        action_contract=actions.action_contract, purpose='qualified_game', synthetic_capacity=False,
        scan=scan, context=context)


def inspect_pruned_game_inference_v3(directory: Path, *, expected_inference_digest: str,
                                    expected_resume_digest: str, original_files: Json, deletion_receipt: Json,
                                    actions: ActionConfigV3, rules: CardInstanceRules,
                                    scan: SealedScan) -> SealedArtifact:
    from dataclasses import asdict

    from gf_common.domain import digest
    input_binding_game_v3(actions.action_contract, rules)
    context = _SealedContextV3(layout_for_contract(actions.action_contract), actions.action_contract,
        rules, 'qualified_game', False, digest(asdict(actions)), True)
    return inspect_pruned_inference(directory, expected_inference_digest=expected_inference_digest,
        expected_resume_digest=expected_resume_digest, original_files=original_files,
        deletion_receipt=deletion_receipt, action_contract=actions.action_contract,
        purpose='qualified_game', synthetic_capacity=False, scan=scan, _context=context)


def _game_sealed_context_v4(actions: ActionConfigV3, rules: CardInstanceRules, *,
                             training_recipe: Json | None = None,
                             optimizer_recipe: Json | None = None) -> _SealedContextV4:
    from dataclasses import asdict

    from gf_common.domain import digest
    input_binding_game_v3(actions.action_contract, rules)
    if type(actions) is not ActionConfigV3 or actions.max_sequence_length > 18:
        raise ContractError("sealed model4 requires exact real18 actions")
    return _SealedContextV4(layout_for_contract(actions.action_contract), actions.action_contract, rules,
        "qualified_game", False, digest(asdict(actions)), True, training_recipe, optimizer_recipe)


def validate_game_inference_metadata_v4(inference: Json, *, actions: ActionConfigV3,
                                        rules: CardInstanceRules) -> Any:
    """Pure recorded format5 validation; no training imports or runtime query."""
    context = _game_sealed_context_v4(actions, rules)
    _, model, _ = _inference_document(inference, actions.action_contract, "qualified_game", False, context)
    return model


def inspect_retained_game_inference_v4(directory: Path, *, expected_inference_digest: str,
                                        actions: ActionConfigV3, rules: CardInstanceRules,
                                        scan: SealedScan) -> SealedArtifact:
    inference = scan.object(directory / "inference.json", checksum(expected_inference_digest))
    validate_game_inference_metadata_v4(inference, actions=actions, rules=rules)
    scan.bundle(directory, inference["weights"])
    scan.finalize()
    return SealedArtifact(inference, None,
        {"inference.json": expected_inference_digest, **inference["weights"]["files"]}, True)


def inspect_retained_game_artifact_v4(directory: Path, *, expected_inference_digest: str,
                                      actions: ActionConfigV3, rules: CardInstanceRules,
                                      training_recipe: Json, optimizer_recipe: Json,
                                      scan: SealedScan) -> SealedArtifact:
    context = _game_sealed_context_v4(actions, rules, training_recipe=training_recipe,
        optimizer_recipe=optimizer_recipe)
    return _inspect_retained_artifact(directory, expected_inference_digest=expected_inference_digest,
        action_contract=actions.action_contract, purpose="qualified_game", synthetic_capacity=False,
        scan=scan, context=context)


def inspect_pruned_game_inference_v4(directory: Path, *, expected_inference_digest: str,
                                    expected_resume_digest: str, original_files: Json, deletion_receipt: Json,
                                    actions: ActionConfigV3, rules: CardInstanceRules,
                                    scan: SealedScan) -> SealedArtifact:
    return inspect_pruned_inference(directory, expected_inference_digest=expected_inference_digest,
        expected_resume_digest=expected_resume_digest, original_files=original_files,
        deletion_receipt=deletion_receipt, action_contract=actions.action_contract,
        purpose="qualified_game", synthetic_capacity=False, scan=scan,
        _context=_game_sealed_context_v4(actions, rules))
