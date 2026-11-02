"""Tree2 schema3 artifacts; trained binding requires an explicit healthy pair."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gf_common.action_contract import ActionContract, parse_action_contract
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
from gf_common.structured_tokens import signature
from gf_models.structured_binding import check_input_binding, input_binding
from gf_models.structured_sealed import SealedScan, checksum
from gf_models.structured_tree_config import (
    MAX_ARRAY_BYTES,
    TREE_RECIPE,
    ConfiguredTreeModelV2,
    TreeDescriptor,
    architecture_config,
    architecture_identity,
    execution_binding,
    feature_width,
    resource_estimate,
)
from gf_models.structured_tree_encoder import categorical_limits, feature_layout
from gf_models.structured_tree_pairing import (
    TrainedProvenance,
    TrainingBinding,
    provenance_for_request,
    validate_request,
)

NATIVE_PARAMETERS = {'objective': 'regression', 'num_class': 1, 'zero_as_missing': False,
                     'linear_tree': False, 'boosting': 'gbdt'}
FIELDS = {'schema_version', 'kind', 'model', 'architecture_identity', 'input_recipe', 'token_signature',
    'feature_recipe', 'feature_signature', 'feature_names', 'categorical_indices', 'semantics',
    'action_config_hash', 'action_contract', 'purpose', 'synthetic_capacity', 'fit_origin', 'provenance',
    'native_parameters', 'runtime', 'model_file', 'tree_descriptor', 'creation_execution'}


def semantics() -> Json:
    return {'catalog': CATALOG_META, 'simulator_contract_revision': SIMULATOR_REVISION,
        'reviewed_simulator_spec_sha256': REVIEWED_SPEC_HASHES, 'policy_input_version': POLICY_INPUT_VERSION,
        'generator_version': GENERATOR_V2, 'reward_policy_version': 'natural-terminal-v1',
        'end_projection_version': END_PROJECTION_VERSION}


def runtime_metadata(*, deadline: float = float('inf')) -> Json:
    _check(deadline)
    # Distribution metadata and owned native-library bytes, not a Booster/Torch import.
    distribution = importlib.metadata.distribution('lightgbm')
    version, numpy_version = distribution.version, importlib.metadata.version('numpy')
    if version != '4.7.0' or numpy_version != '2.5.4':
        raise ContractError('Tree2 requires audited LightGBM4.7.0/NumPy2.5.4')
    files = [path for path in distribution.files or () if str(path).endswith('/lib_lightgbm.so')]
    if len(files) != 1:
        raise ContractError('exact owned LightGBM native library required')
    path = Path(str(distribution.locate_file(files[0])))
    value = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024**2):
            _check(deadline)
            value.update(block)
    _check(deadline)
    return {'numpy_version': numpy_version, 'lightgbm_version': version, 'native_library_sha256': value.hexdigest()}


def _native(*, deadline: float = float('inf')) -> Any:
    actual = runtime_metadata(deadline=deadline)
    _check(deadline)
    module = importlib.import_module('lightgbm')
    _check(deadline)
    if module.__version__ != actual['lightgbm_version']:
        raise ContractError('Tree2 native module/distribution mismatch')
    return module


def _parameters(text: str) -> Json:
    if text.count('\nparameters:\n') != 1 or text.count('\nend of parameters\n') != 1:
        raise ContractError('Tree2 native parameter declaration required')
    body = text.split('\nparameters:\n', 1)[1].split('\nend of parameters\n', 1)[0]
    result = {}
    for line in body.splitlines():
        match = re.fullmatch(r'\[([A-Za-z0-9_]+): (.*)\]', line)
        if match is None or match[1] in result:
            raise ContractError('duplicate/malformed native parameter declaration')
        result[match[1]] = match[2]
    expected = {'objective': 'regression', 'num_class': '1', 'zero_as_missing': '0', 'linear_tree': '0',
                'boosting': 'gbdt', 'device_type': 'cpu'}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ContractError('unsupported actual Tree2 objective/missing/linear/boosting semantics')
    return result


def _text_descriptor(text: str, model: Any, model_hash: str, deadline: float = float('inf'), *,
                     split_semantics: list[list[tuple[int, bool, str, bool]]] | None = None) -> TreeDescriptor:
    _check(deadline)
    params = _parameters(text)
    if params.get('num_leaves') != str(model.network.num_leaves):
        raise ContractError('Tree2 actual native architecture parameter mismatch')
    head = text.split('\nTree=', 1)[0]
    lines = {}
    for line in head.splitlines()[1:]:
        if line:
            key, separator, value = line.partition('=')
            if not separator or key in lines:
                raise ContractError('Tree2 malformed native model header')
            lines[key] = value
    layout = feature_layout(model.encoder)
    names = layout['feature_names']
    categorical = set(layout['categorical_indices'])
    if (lines.get('version') != 'v4' or lines.get('num_class') != '1'
            or lines.get('num_tree_per_iteration') != '1' or lines.get('objective') != 'regression'
            or lines.get('feature_names', '').split(' ') != names
            or lines.get('max_feature_idx') != str(len(names)-1)):
        raise ContractError('Tree2 actual native model/header feature mismatch')
    declared = params.get('categorical_feature', '')
    if declared != ','.join(str(i) for i in layout['categorical_indices']):
        raise ContractError('Tree2 actual category declarations mismatch, including unused columns')
    # The original model format stores each node's children and leaf count.
    # Traverse indices without importing/executing native code; detect cycles,
    # malformed ranges and duplicate unreachable branches explicitly.
    trees = re.split(r'\nTree=[0-9]+\n', text)[1:]
    if not trees:
        raise ContractError('Tree2 nonempty scalar model required')
    deepest = 0
    for tree in trees:
        _check(deadline)
        tree = tree.split('\n\n', 1)[0]
        values = dict(line.split('=', 1) for line in tree.splitlines() if '=' in line)
        try:
            leaves = int(values['num_leaves'])
            if not 1 <= leaves <= model.network.num_leaves or values.get('is_linear') != '0':
                raise ValueError('unsupported leaf structure')
            left = [int(v) for v in values.get('left_child', '').split()]
            right = [int(v) for v in values.get('right_child', '').split()]
            if len(left) != leaves - 1 or len(right) != leaves - 1:
                raise ValueError('child shape')
            nodes, leaf_ids, stack = set(), set(), [(0, 0)] if leaves > 1 else [(-1, 0)]
            while stack:
                _check(deadline)
                node, depth = stack.pop()
                deepest = max(deepest, depth)
                if node < 0:
                    leaf = -1-node
                    if leaf >= leaves or leaf in leaf_ids:
                        raise ValueError('leaf range/reuse')
                    leaf_ids.add(leaf)
                else:
                    if node >= leaves-1 or node in nodes:
                        raise ValueError('node range/reuse')
                    nodes.add(node)
                    stack.extend(((left[node], depth+1), (right[node], depth+1)))
            if len(nodes) != leaves-1 or len(leaf_ids) != leaves:
                raise ValueError('unreachable node')
            decisions = [int(v) for v in values.get('decision_type', '').split()]
            features = [int(v) for v in values.get('split_feature', '').split()]
            thresholds = values.get('threshold', '').split()
            if len(decisions) != leaves-1 or len(features) != leaves-1 or len(thresholds) != leaves-1:
                raise ValueError('split feature/decision/threshold count')
            splits = []
            num_cat = int(values['num_cat'])
            if not 0 <= num_cat <= leaves-1 or num_cat != sum(bool(v & 1) for v in decisions):
                raise ValueError('categorical split count')
            for feature, decision, threshold in zip(features, decisions, thresholds, strict=True):
                if (not 0 <= feature < len(names) or not 0 <= decision <= 15
                        or bool(decision & 1) != (feature in categorical)):
                    raise ValueError('split feature range or categorical/numeric kind')
                missing = (decision >> 2) & 3
                if missing not in (0, 2):
                    raise ValueError('unsupported or zero-missing split flag')
                if decision & 1:
                    if not 0 <= int(threshold) < num_cat:
                        raise ValueError('categorical threshold index')
                elif not math.isfinite(float(threshold)):
                    raise ValueError('nonfinite numeric threshold')
                splits.append((feature, bool(decision & 1), 'NaN' if missing == 2 else 'None', bool(decision & 2)))
            if split_semantics is not None:
                split_semantics.append(splits)
            leaf_values = [float(v) for v in values['leaf_value'].split()]
            if len(leaf_values) != leaves or not all(math.isfinite(v) for v in leaf_values):
                raise ValueError('nonfinite/scalar leaves')
        except (KeyError, ValueError, OverflowError) as error:
            raise ContractError('invalid actual Tree2 scalar tree structure') from error
    return TreeDescriptor(len(trees), deepest, model_hash, len(text.encode()))


@dataclass(frozen=True)
class TreeOrigin:
    action_config_hash: str
    action_contract: ActionContract
    purpose: str
    synthetic_capacity: bool
    fit_origin: str = 'toy'
    provenance: TrainedProvenance | None = None

    def __post_init__(self) -> None:
        checksum(self.action_config_hash)
        input_binding(self.action_contract, self.purpose, self.synthetic_capacity)
        if self.fit_origin=='toy':
            if self.provenance is not None or self.purpose!='component':raise ContractError('toy Tree2 requires null/component origin')
        elif self.fit_origin=='trained':
            if type(self.provenance) is not TrainedProvenance:raise ContractError('immutable trained Tree2 provenance required')
            if canonical(self.provenance.to_dict()['input_binding'])!=canonical(input_binding(self.action_contract,self.purpose,self.synthetic_capacity)):
                raise ContractError('trained Tree2 input provenance cannot be relabeled')
        else:raise ContractError('unsupported Tree2 fit origin')

    def fields(self)->Json:
        return {'action_config_hash':self.action_config_hash,**input_binding(self.action_contract,self.purpose,self.synthetic_capacity),
                'fit_origin':self.fit_origin,'provenance':None if self.provenance is None else self.provenance.to_dict()}


@dataclass(frozen=True)
class LoadedTree:
    booster: Any
    model: ConfiguredTreeModelV2
    origin: TreeOrigin
    descriptor: TreeDescriptor
    training_binding: TrainingBinding | None = None
    creation_execution: bytes | None = None

    def __post_init__(self) -> None:
        _bind_origin(self.booster, self.origin)


def _bind_origin(booster: Any, origin: TreeOrigin, *, fresh: bool = False) -> None:
    if type(origin) is not TreeOrigin:
        raise ContractError('immutable Tree2 origin required')
    previous = getattr(booster, '_structured_tree_origin', None)
    if previous is not None:
        if type(previous) is not TreeOrigin or previous != origin:
            raise ContractError('Tree2 actual native model origin cannot be relabeled')
    elif fresh:
        booster._structured_tree_origin = origin
    else:
        raise ContractError('Tree2 native model has no validated original binding')


def inspect_artifact(directory: Path, *, expected_digest: str, model: ConfiguredTreeModelV2,
                     action_config_hash: str, action_contract: ActionContract,
                     purpose: str, synthetic_capacity: bool, deadline: float = float('inf'),
                     expected_tree_descriptor: TreeDescriptor | None = None,
                     expected_feature_signature: str | None = None,
                     _scan: SealedScan | None = None) -> tuple[Json, str]:
    checksum(expected_digest)
    checksum(action_config_hash)
    if expected_tree_descriptor is not None and type(expected_tree_descriptor) is not TreeDescriptor:
        raise ContractError('closed expected Tree2 descriptor required')
    if expected_feature_signature is not None:
        checksum(expected_feature_signature)
    scan = _scan if _scan is not None else SealedScan(max_owned_bytes=model.resources.max_owned_bytes, deadline=deadline)
    scan.check()
    scan.inventory(directory, {'inference.json', 'model.txt'})
    manifest = scan.object(directory / 'inference.json', expected_digest)
    closed(manifest, FIELDS)
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 3 or manifest['kind'] != 'structured-tree-inference':
        raise ContractError('unsupported Tree2 artifact schema/kind')
    classification = input_binding(action_contract, purpose, synthetic_capacity)
    check_input_binding({key: manifest[key] for key in classification}, action_contract=action_contract,
                        purpose=purpose, synthetic_capacity=synthetic_capacity)
    TreeOrigin(action_config_hash, action_contract, purpose, synthetic_capacity, manifest['fit_origin'],
               None if manifest['provenance'] is None else TrainedProvenance.parse(manifest['provenance']))
    layout = feature_layout(model.encoder)
    if expected_feature_signature is not None and manifest['feature_signature'] != expected_feature_signature:
        raise ContractError('Tree2 inspected feature signature differs from own admitted binding')
    for key, expected in {'model': asdict(model.architecture()), 'architecture_identity': architecture_identity(model.architecture()),
            'input_recipe': INPUT_RECIPE, 'token_signature': signature(model.encoder.aggregate_window),
            'feature_recipe': TREE_RECIPE, 'feature_signature': layout['feature_signature'],
            'feature_names': layout['feature_names'], 'categorical_indices': layout['categorical_indices'],
            'semantics': semantics(), 'action_config_hash': action_config_hash, 'native_parameters': NATIVE_PARAMETERS}.items():
        if canonical(manifest[key]) != canonical(expected):
            raise ContractError('Tree2 artifact derived declaration mismatch: ' + key)
    architecture_config(manifest['model'])
    closed(manifest['runtime'], {'numpy_version', 'lightgbm_version', 'native_library_sha256'})
    checksum(manifest['runtime']['native_library_sha256'])
    if manifest['runtime']['numpy_version'] != '2.5.4' or manifest['runtime']['lightgbm_version'] != '4.7.0':
        raise ContractError('unsupported recorded Tree2 runtime')
    creation = closed(manifest['creation_execution'], {'inference', 'resources', 'tree_descriptor'})
    descriptor = TreeDescriptor(**closed(manifest['tree_descriptor'], set(TreeDescriptor.__dataclass_fields__)))
    if expected_tree_descriptor is not None and descriptor != expected_tree_descriptor:
        raise ContractError('Tree2 inspected descriptor differs from own admitted binding')
    if canonical(creation['tree_descriptor']) != canonical(asdict(descriptor)):
        raise ContractError('Tree2 creation/actual tree descriptor mismatch')
    from gf_models.structured_tree_config import configured_tree_v2
    created_model = configured_tree_v2({**manifest['model'], 'checkpoint': None,
                                      'inference': creation['inference'], 'resources': creation['resources']})
    resource_estimate(created_model, model_bytes=descriptor.model_bytes)
    file = closed(manifest['model_file'], {'file', 'sha256', 'bytes'})
    integer(file['bytes'], 1, MAX_ARRAY_BYTES)
    checksum(file['sha256'])
    if file['file'] != 'model.txt' or descriptor.model_sha256 != file['sha256'] or descriptor.model_bytes != file['bytes']:
        raise ContractError('Tree2 owned model-file binding mismatch')
    resource_estimate(model, model_bytes=file['bytes'], extra_bytes=scan.retained)
    with scan.file(directory / 'model.txt', file['sha256']) as (source, info):
        if info.st_size != file['bytes']:
            raise ContractError('Tree2 native model file length changed')
        scan.charge(16 * info.st_size, temporary=True)
        text = source.read(info.st_size).decode('utf-8')
        if _text_descriptor(text, model, file['sha256'], deadline) != descriptor:
            raise ContractError('Tree2 actual model tree count/depth mismatch')
    scan.finalize()
    return manifest, text


def _validate_booster(booster: Any, model: ConfiguredTreeModelV2, *, deadline: float = float('inf')) -> tuple[str, TreeDescriptor]:
    _check(deadline)
    text = booster.model_to_string(num_iteration=-1)
    model_hash = hashlib.sha256(text.encode()).hexdigest()
    splits: list[list[tuple[int, bool, str, bool]]] = []
    descriptor = _text_descriptor(text, model, model_hash, deadline, split_semantics=splits)
    _check(deadline)
    dump = booster.dump_model(num_iteration=-1)
    _check(deadline)
    layout = feature_layout(model.encoder)
    if (booster.num_feature() != feature_width(model.encoder.max_positions) or booster.feature_name() != layout['feature_names']
            or dump['num_class'] != 1 or dump['objective'] != 'regression' or dump['average_output']
            or booster.num_trees() != descriptor.tree_count):
        raise ContractError('actual Tree2 Booster semantics/features mismatch')
    actual_cats = [i for i, name in enumerate(layout['feature_names']) if dump['feature_infos'].get(name, {}).get('values')]
    if actual_cats != layout['categorical_indices']:
        raise ContractError('actual Tree2 Booster unused category declarations mismatch')
    if len(dump['tree_info']) != len(splits):
        raise ContractError('actual Tree2 Booster split/tree count mismatch')
    for tree, expected_splits in zip(dump['tree_info'], splits, strict=True):
        observed = {}
        todo = [tree['tree_structure']]
        while todo:
            _check(deadline)
            node = todo.pop()
            if 'split_index' not in node:
                continue
            index, feature = node['split_index'], node['split_feature']
            if (type(index) is not int or not 0 <= index < len(expected_splits) or index in observed
                    or type(feature) is not int or not 0 <= feature < len(layout['feature_names'])
                    or node['decision_type'] not in ('==', '<=') or type(node['default_left']) is not bool):
                raise ContractError('actual Tree2 Booster invalid split index/feature/kind')
            observed[index] = (feature, node['decision_type'] == '==', node['missing_type'], node['default_left'])
            todo.extend((node['left_child'], node['right_child']))
        if len(observed) != len(expected_splits) or any(observed[i] != expected for i, expected in enumerate(expected_splits)):
            raise ContractError('actual Tree2 Booster per-node categorical/missing semantics mismatch')
    for key, expected in NATIVE_PARAMETERS.items():
        if type(booster.params.get(key)) is not type(expected) or booster.params[key] != expected:
            raise ContractError('actual Tree2 Booster semantic parameters mismatch: ' + key)
    _check(deadline)
    return text, descriptor


def load_artifact(directory: Path, **options: Any) -> LoadedTree:
    training_directory=options.pop('training_directory',None)
    training_binding=options.pop('training_binding',None)
    manifest, text = inspect_artifact(directory, **options)
    deadline = options.get('deadline', float('inf'))
    pair=None
    if manifest['fit_origin']=='trained':
        if training_directory is None:raise ContractError('trained Tree2 requires explicit complete paired training directory')
        from gf_models.structured_tree_pairing import inspect_training
        pair=inspect_training(training_directory,expected_binding=training_binding,
            max_owned_bytes=options['model'].resources.max_owned_bytes,deadline=deadline)
        if Path(training_directory).absolute()/'artifact'!=Path(directory).absolute() or canonical(pair.inference)!=canonical(manifest):
            raise ContractError('Tree2 explicit training/artifact pair differs')
    elif training_directory is not None or training_binding is not None:
        raise ContractError('toy Tree2 cannot claim a trained binding')
    if manifest['runtime'] != runtime_metadata(deadline=deadline):
        raise ContractError('actual Tree2 native runtime differs from recorded identity')
    model = options['model']
    resource_estimate(model,model_bytes=manifest['tree_descriptor']['model_bytes'],extra_bytes=0 if pair is None else pair.scan.retained)
    booster = _native(deadline=deadline).Booster(model_str=text)
    _check(deadline)
    _, actual = _validate_booster(booster, model, deadline=deadline)
    if actual != TreeDescriptor(**manifest['tree_descriptor']):
        raise ContractError('loaded Tree2 Booster descriptor mismatch')
    origin = TreeOrigin(options['action_config_hash'], options['action_contract'], options['purpose'], options['synthetic_capacity'],
        manifest['fit_origin'],None if manifest['provenance'] is None else TrainedProvenance.parse(manifest['provenance']))
    _bind_origin(booster, origin, fresh=True)
    _check(deadline)
    if pair is not None:pair.finalize()
    return LoadedTree(booster, model, origin, actual,None if pair is None else pair.binding,
        None if pair is None else canonical(manifest['creation_execution']).encode())


def _check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError('Tree2 artifact/runtime deadline')


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _written(directory: Path, directory_fd: int, name: str, fd: int,
             snapshot: tuple[int, ...], expected: str, deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError('Tree2 publication deadline')
    value = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while block := os.read(fd, 1024**2):
        if time.monotonic() >= deadline:
            raise TimeoutError('Tree2 publication deadline')
        value.update(block)
    if value.hexdigest() != expected:
        raise ContractError('Tree2 output hash changed from owned written snapshot')
    _post_written(directory, directory_fd, name, fd, snapshot)


def _post_written(directory: Path, directory_fd: int, name: str, fd: int,
                  snapshot: tuple[int, ...]) -> None:
    held, current = os.fstat(fd), os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    path = (directory / name).lstat()
    if not all(stat.S_ISREG(item.st_mode) and _identity(item) == snapshot for item in (held, current, path)):
        raise ContractError('Tree2 output changed from owned written snapshot')


def _publish(directory: Path, directory_fd: int, name: str, payload: bytes,
             deadline: float) -> tuple[int, tuple[int, ...], str]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(os.dup(fd), 'wb') as output:
            output.write(payload)
            output.flush()
        snapshot = _identity(os.fstat(fd))
        sha = hashlib.sha256(payload).hexdigest()
        _written(directory, directory_fd, name, fd, snapshot, sha, deadline)
        return fd, snapshot, sha
    except BaseException:
        os.close(fd)
        raise


def save_artifact(directory: Path, tree: LoadedTree, *, deadline: float = float('inf')) -> Json:
    if type(tree) is LoadedTree and tree.origin.fit_origin=='trained':
        raise ContractError('trained Tree2 resave requires exact paired-bundle copy; creation/completion cannot be rewritten')
    return _save_inference(directory,tree,deadline=deadline)


def _save_inference(directory: Path, tree: LoadedTree, *, deadline: float = float('inf')) -> Json:
    if type(tree) is not LoadedTree or type(tree.origin) is not TreeOrigin:
        raise ContractError('Tree2 immutable actual loaded origin required')
    _bind_origin(tree.booster, tree.origin)
    text, descriptor = _validate_booster(tree.booster, tree.model, deadline=deadline)
    if descriptor != tree.descriptor:
        raise ContractError('Tree2 actual wrapper/Booster descriptor changed')
    encoded = text.encode()
    resource_estimate(tree.model, model_bytes=len(encoded))
    if time.monotonic() >= deadline:
        raise TimeoutError('Tree2 save deadline')
    layout = feature_layout(tree.model.encoder)
    manifest = {'schema_version': 3, 'kind': 'structured-tree-inference', 'model': asdict(tree.model.architecture()),
        'architecture_identity': architecture_identity(tree.model.architecture()), 'input_recipe': INPUT_RECIPE,
        'token_signature': signature(tree.model.encoder.aggregate_window), 'feature_recipe': TREE_RECIPE,
        'feature_signature': layout['feature_signature'], 'feature_names': layout['feature_names'],
        'categorical_indices': layout['categorical_indices'], 'semantics': semantics(),
        **tree.origin.fields(), 'native_parameters': NATIVE_PARAMETERS, 'runtime': runtime_metadata(deadline=deadline),
        'model_file': {'file': 'model.txt', 'sha256': descriptor.model_sha256, 'bytes': len(encoded)},
        'tree_descriptor': asdict(descriptor), 'creation_execution': execution_binding(tree.model, descriptor) if tree.creation_execution is None else json.loads(tree.creation_execution)}
    metadata = (canonical(manifest)+'\n').encode()
    resource_estimate(tree.model, model_bytes=len(encoded), extra_bytes=32 * len(metadata) + 65536)
    parents = {}
    for parent in (directory.parent, *directory.parent.parents):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ContractError('Tree2 owned output parent must be nonsymlink directory')
        parents[parent] = (info.st_dev, info.st_ino)
    directory.mkdir(mode=0o700, exist_ok=False)
    created = directory.lstat()
    from contextlib import ExitStack
    with ExitStack() as handles:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        handles.callback(os.close, directory_fd)
        def check_directory() -> None:
            held, current = os.fstat(directory_fd), directory.lstat()
            if any(not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (created.st_dev, created.st_ino)
                   for info in (held, current)):
                raise ContractError('Tree2 output directory ownership changed')
            for parent, expected in parents.items():
                info = parent.lstat()
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != expected:
                    raise ContractError('Tree2 output parent ownership changed')
        check_directory()
        outputs = []
        for name, payload in (('model.txt', encoded), ('inference.json', metadata)):
            check_directory()
            fd, snapshot, sha = _publish(directory, directory_fd, name, payload, deadline)
            handles.callback(os.close, fd)
            outputs.append((name, fd, snapshot, sha))
        for name, fd, snapshot, sha in outputs:
            _written(directory, directory_fd, name, fd, snapshot, sha, deadline)
        for name, fd, snapshot, _ in outputs:
            _check(deadline)
            _post_written(directory, directory_fd, name, fd, snapshot)
        check_directory()
        _check(deadline)
    return manifest


def fit_toy(blocks: list[Any], labels: Any, validation_blocks: list[Any], validation_labels: Any, *,
            model: ConfiguredTreeModelV2, origin: TreeOrigin, seed: int, rounds: int = 500,
            early_stopping: int = 50, learning_rate: float = .05, min_data_in_leaf: int = 20,
            deadline: float = float('inf')) -> LoadedTree:
    """Explicit component-only categorical API test; never source/fit authority."""
    import numpy as np

    integer(seed, 0, 2**31-1)
    integer(rounds, 1, 500)
    integer(early_stopping, 1, 2**31-1)
    integer(min_data_in_leaf, 1, 2**31-1)
    if type(origin) is not TreeOrigin or origin.fit_origin!='toy' or origin.provenance is not None or type(learning_rate) not in (int, float) or not 0 < learning_rate <= 1:
        raise ContractError('explicit toy Tree2 fit origin/settings required')
    if not blocks or not validation_blocks:
        raise ContractError('nonempty admitted Tree2 matrix lists required')
    width, raw_bytes = feature_width(model.encoder.max_positions), 0
    for matrices, targets in ((blocks, labels), (validation_blocks, validation_labels)):
        count = 0
        for matrix in matrices:
            if (type(matrix) is not np.ndarray or matrix.dtype != np.float32 or matrix.ndim != 2
                    or matrix.shape[1] != width or not matrix.shape[0] or not matrix.flags.c_contiguous
                    or matrix.nbytes > model.resources.max_array_bytes):
                raise ContractError('bounded contiguous exact-width FP32 Tree2 blocks required')
            raw_bytes += matrix.nbytes
            count += matrix.shape[0]
        if type(targets) is not np.ndarray or targets.dtype != np.float32 or targets.shape != (count,):
            raise ContractError('Tree2 finite FP32 labels remain separate from rows')
        raw_bytes += targets.nbytes
    # Admit caller-owned blocks/labels, native Dataset copies and workspace as a
    # joint soft estimate before masks, Dataset construction or Booster creation.
    resource_estimate(model, extra_bytes=5 * raw_bytes)
    layout = feature_layout(model.encoder)
    for matrices, targets in ((blocks, labels), (validation_blocks, validation_labels)):
        if not np.isfinite(targets).all():
            raise ContractError('nonfinite Tree2 labels')
        for matrix in matrices:
            for start in range(0, len(matrix), model.inference.candidate_chunk):
                chunk = matrix[start:start+model.inference.candidate_chunk]
                if not np.isfinite(chunk).all():
                    raise ContractError('nonfinite Tree2 features')
                for index, maximum in zip(layout['categorical_indices'], categorical_limits(model.encoder), strict=True):
                    values = chunk[:, index]
                    if np.any(values < 0) or np.any(values > maximum) or np.any(values != np.floor(values)):
                        raise ContractError('Tree2 categories require exact closed integer codes')
    from gf_common.domain import digest
    derived = int(digest({'recipe': model.initialization.recipe, 'model_seed': seed})[:8], 16) % (2**31-1)
    params = {**NATIVE_PARAMETERS, 'metric': 'l2', 'learning_rate': learning_rate,
        'num_leaves': model.network.num_leaves, 'min_data_in_leaf': min_data_in_leaf,
        'num_threads': model.inference.num_threads, 'device_type': 'cpu', 'verbosity': -1,
        'deterministic': True, 'force_col_wise': True, 'feature_pre_filter': False,
        'seed': derived, 'data_random_seed': derived, 'feature_fraction_seed': derived, 'bagging_seed': derived}
    native = _native(deadline=deadline)
    def check(_: Any = None) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError('Tree2 toy native fit deadline')
    check()
    train = native.Dataset(blocks, label=labels, feature_name=layout['feature_names'],
        categorical_feature=layout['categorical_indices'], params=params, free_raw_data=True)
    validation = native.Dataset(validation_blocks, label=validation_labels, reference=train,
        feature_name=layout['feature_names'], categorical_feature=layout['categorical_indices'], params=params, free_raw_data=True)
    booster = native.train(params, train, num_boost_round=rounds, valid_sets=[validation],
        callbacks=[check, native.early_stopping(early_stopping, verbose=False)])
    check()
    _, descriptor = _validate_booster(booster, model, deadline=deadline)
    _bind_origin(booster, origin, fresh=True)
    return LoadedTree(booster, model, origin, descriptor)


def binding_options(directory: Path, *, expected_digest: str, model: ConfiguredTreeModelV2,
                    action_config_hash: str, action_contract: ActionContract, purpose: str,
                    synthetic_capacity: bool, deadline: float = float('inf')) -> Json:
    """Parent-owned exact metadata → closed own-policy DTO, without native load."""
    from gf_models.structured_tree_pairing import resolve_checkpoint
    directory,resolved_digest,training=resolve_checkpoint(directory,max_owned_bytes=model.resources.max_owned_bytes,deadline=deadline)
    if expected_digest!=resolved_digest:raise ContractError('Tree2 resolved checkpoint manifest differs from expected identity')
    manifest, _ = inspect_artifact(directory, expected_digest=expected_digest, model=model,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose=purpose,
        synthetic_capacity=synthetic_capacity, deadline=deadline)
    from gf_models.structured_tree_encoder import safe_tree_options_v2
    value = {'version': '2', 'checkpoint': str(directory), 'expected_manifest_digest': expected_digest,
        'model': manifest['model'], 'inference': asdict(model.inference), 'resources': asdict(model.resources),
        'tree_descriptor': manifest['tree_descriptor'], 'feature_signature': manifest['feature_signature'],
        **input_binding(action_contract, purpose, synthetic_capacity)}
    if training is not None:value['training_binding']=asdict(training)
    return safe_tree_options_v2(value, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)


def save_provisional_trained(directory:Path,booster:Any,model:ConfiguredTreeModelV2,request:Json,request_sha256:str,
                             action_config_hash:str,*,deadline:float)->Json:
    """Worker-only structural stage; never a bindable completed training pair."""
    validate_request(request);checksum(request_sha256);checksum(action_config_hash)
    if canonical(request['model'])!=canonical(asdict(model.architecture())) or canonical(request['execution'])!=canonical({'inference':asdict(model.inference),'resources':asdict(model.resources)}):
        raise ContractError('provisional Tree2 actual request/model execution mismatch')
    classification=request['input_binding'];contract=parse_action_contract(classification['action_contract'])
    origin=TreeOrigin(action_config_hash,contract,classification['purpose'],classification['synthetic_capacity'],'trained',provenance_for_request(request,request_sha256))
    _,descriptor=_validate_booster(booster,model,deadline=deadline)
    if descriptor.model_bytes>request['model_envelope']['text_bytes']:raise ContractError('actual native Tree2 exceeds pre-admitted model envelope')
    _bind_origin(booster,origin,fresh=True)
    tree=LoadedTree(booster,model,origin,descriptor,creation_execution=canonical({**request['execution'],'tree_descriptor':asdict(descriptor)}).encode())
    return _save_inference(directory,tree,deadline=deadline)
