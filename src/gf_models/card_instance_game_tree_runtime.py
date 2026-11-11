"""Qualified Tree3 own-policy execution metadata and controller artifact custody.

Only the controller's retained pair validates training completion. Child metadata
contains opaque digest references and cannot supply Source or fitting authority.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_models.structured_binding import input_binding_game_v3
from gf_models.structured_sealed import basename, checksum
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV3,
    TreeDescriptor,
    architecture_config_v3,
    configured_tree_v3,
    numeric_costs_v3,
)
from gf_models.structured_tree_encoder import feature_layout_v3

TREE_INPUT_CONTRACT = 'player-card-refs-v2'
TREE_OPTIONS_FIELDS = {'version', 'checkpoint', 'expected_manifest_digest', 'model', 'inference',
    'resources', 'action_config_hash', 'tree_descriptor', 'feature_signature', 'source_kind',
    'source_origin', 'training_pair'}


def _configured(options: Json) -> ConfiguredTreeModelV3:
    architecture = architecture_config_v3(options['model'])
    return configured_tree_v3({**asdict(architecture), 'checkpoint': options['checkpoint'],
        'inference': options['inference'], 'resources': options['resources']})


def safe_tree_options_game_v3(value: Any, *, actions: ActionConfigV3, rules: CardInstanceRules) -> Json:
    """Closed, pure execution declaration; no external I/O or native allocation."""
    from gf_common.card_instance_game_tree import PACKED_SOURCE_KIND
    packed = isinstance(value, dict) and value.get('source_kind') == PACKED_SOURCE_KIND
    raw = closed(value, TREE_OPTIONS_FIELDS | ({'storage_binding'} if packed else set()))
    input_binding_game_v3(actions.action_contract, rules)
    if raw['version'] != '3' or type(raw['checkpoint']) is not str or not raw['checkpoint']:
        raise ContractError('explicit own Tree3 checkpoint required')
    checksum(raw['expected_manifest_digest'])
    if checksum(raw['action_config_hash']) != digest(asdict(actions)):
        raise ContractError('Tree3 action configuration differs')
    model = _configured(raw)
    if (actions.max_sequence_length > 18 or model.encoder.max_positions > 18
            or actions.max_sequence_length > model.encoder.max_positions
            or actions.max_candidates > model.inference.max_candidates
            or model.inference.candidate_chunk != 256):
        raise ContractError('Tree3 real18/chunk256/action capacity mismatch')
    descriptor = TreeDescriptor(**closed(raw['tree_descriptor'], set(TreeDescriptor.__dataclass_fields__)))
    from gf_models.structured_tree_config import resource_estimate
    resource_estimate(model, model_bytes=descriptor.model_bytes)
    if checksum(raw['feature_signature']) != feature_layout_v3(model.encoder)['feature_signature']:
        raise ContractError('Tree3 exact feature recipe required')
    pair = closed(raw['training_pair'], {'request_sha256', 'completion_sha256'})
    for pair_digest in pair.values():
        checksum(pair_digest)
    # The GAME record context owns source-origin semantics; a runtime declaration
    # validates their shape only and never grants source/fit permission.
    from gf_models.structured_tree_pairing import game_records
    game_records(raw)
    return clone(raw)


def policy_label_tree_game_v3(options: Json, *, actions: ActionConfigV3, rules: CardInstanceRules) -> str:
    safe = safe_tree_options_game_v3(options, actions=actions, rules=rules)
    execution = {key: value for key, value in safe.items() if key != 'checkpoint'}
    return 'lightgbm-tree-v3-' + safe['expected_manifest_digest'] + '-' + digest(execution)


def game_tree_settings(options: Json, *, actions: ActionConfigV3, rules: CardInstanceRules) -> Any:
    from gf_models.structured_runtime import WorkerCaps
    safe = safe_tree_options_game_v3(options, actions=actions, rules=rules)
    model = _configured(safe)
    caps = WorkerCaps(max_positions=actions.max_sequence_length, max_candidates=actions.max_candidates)
    return caps, numeric_costs_v3(model, TreeDescriptor(**safe['tree_descriptor'])), 0, 0


def validate_tree_metadata(raw: Any, *, actions: ActionConfigV3, rules: CardInstanceRules) -> tuple[Json, str]:
    """Validate safe recorded own artifacts without opening Source or original paths."""
    closed(raw, {'input_contract', 'policy_options', 'inference_manifest', 'files'})
    if raw['input_contract'] != TREE_INPUT_CONTRACT:
        raise ContractError('Tree3 corrected player-card-refs-v2 metadata required')
    safe = safe_tree_options_game_v3(raw['policy_options'], actions=actions, rules=rules)
    model = _configured(safe)
    from gf_models.structured_tree_artifacts import validate_game_inference_metadata_v3
    validate_game_inference_metadata_v3(raw['inference_manifest'], model=model,
        action_config_hash=safe['action_config_hash'], expected_tree_descriptor=TreeDescriptor(**safe['tree_descriptor']),
        expected_feature_signature=safe['feature_signature'])
    manifest = raw['inference_manifest']
    if (manifest['fit_origin'] != 'trained' or canonical(manifest['source_origin']) != canonical(safe['source_origin'])
            or manifest['source_kind'] != safe['source_kind']
            or canonical(manifest.get('storage_binding')) != canonical(safe.get('storage_binding'))
            or manifest['provenance']['request_sha256'] != safe['training_pair']['request_sha256']):
        raise ContractError('Tree3 own inference request/source binding mismatch')
    expected = {'inference.json': safe['expected_manifest_digest'], 'model.txt': manifest['model_file']['sha256']}
    files = raw['files']
    if type(files) is not list or len(files) != 2:
        raise ContractError('Tree3 exact own artifact file inventory required')
    names = []
    for row in files:
        closed(row, {'path', 'sha256', 'identity'})
        name = basename(row['path'])
        if name not in expected or checksum(row['sha256']) != expected[name]:
            raise ContractError('Tree3 own artifact file digest mismatch')
        names.append(name)
        if type(row['identity']) not in (list, tuple) or len(row['identity']) != 5:
            raise ContractError('Tree3 five-integer own file identity required')
        for value in row['identity']:
            integer(value, 0)
    if names != sorted(expected):
        raise ContractError('Tree3 file inventory must be exact sorted own basenames')
    return safe, policy_label_tree_game_v3(safe, actions=actions, rules=rules)


class RetainedGamePolicyArtifacts:
    """Controller-only custody of actual qualified pairs and optional NN originals."""
    def __init__(self, models: tuple[Any, ...], actions: ActionConfigV3, *, rules: CardInstanceRules,
                 max_owned_bytes: int, preflight_deadline: float, finalization_deadline: float) -> None:
        import math
        from pathlib import Path

        from gf_models.card_instance_artifacts import RetainedGameArtifacts
        from gf_models.structured_binding import ConfiguredNeuralModelV3
        from gf_models.structured_tree_pairing import (
            inspect_training_game_v3,
            resolve_checkpoint_game_v3,
        )
        if (type(models) is not tuple or not 1 <= len(models) <= 2
                or any(type(model) not in (ConfiguredNeuralModelV3, ConfiguredTreeModelV3) for model in models)):
            raise ContractError('one or two closed learned game policy roles required')
        for deadline in (preflight_deadline, finalization_deadline):
            if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
                raise ContractError('original finite game artifact deadlines required')
        if preflight_deadline > finalization_deadline:
            raise ContractError('artifact preflight exceeds original session deadline')
        integer(max_owned_bytes, 1, 2**63-1)
        input_binding_game_v3(actions.action_contract, rules)
        if actions.max_sequence_length > 18 or any(model.checkpoint is None for model in models):
            raise ContractError('game inference requires explicit real18 checkpoints')
        self._actions, self._rules = actions, rules
        self._deadline = finalization_deadline
        self._limit = max_owned_bytes
        self._released = self._failed = False
        self._pairs: list[Any] = []
        self._workers: list[Any] = []
        self._entries: dict[str, Json] = {}
        self._labels: dict[str, str] = {}
        self._neural: RetainedGameArtifacts | None = None
        # Each retained external owner has a disjoint share of the one controller
        # budget. Execution overrides cannot increase this custody allocation.
        share = max_owned_bytes // len(models)
        try:
            neural = tuple(model for model in models if type(model) is ConfiguredNeuralModelV3)
            if neural:
                self._neural = RetainedGameArtifacts(neural, actions, rules=rules,
                    max_owned_bytes=share * len(neural), preflight_deadline=preflight_deadline,
                    finalization_deadline=finalization_deadline)
            for model in models:
                if type(model) is not ConfiguredTreeModelV3:
                    continue
                assert model.checkpoint is not None
                directory, sha, binding = resolve_checkpoint_game_v3(Path(model.checkpoint),
                    max_owned_bytes=share, deadline=preflight_deadline)
                if binding is None:
                    raise ContractError('Tree game inference requires an actual healthy trained GAME pair')
                pair = inspect_training_game_v3(Path(binding.directory), expected_binding=binding,
                    max_owned_bytes=share, deadline=preflight_deadline)
                self._pairs.append(pair)  # Retain before any later fallible inspection.
                manifest = pair.inference
                if canonical(manifest['model']) != canonical(asdict(model.architecture())):
                    raise ContractError('Tree actual trained architecture differs from Config')
                from gf_models.structured_tree_config import resource_estimate
                from gf_models.structured_tree_encoder import _owned
                descriptor_bytes = manifest['tree_descriptor']['model_bytes']
                pair.scan.charge(34*_owned(manifest)+65536, temporary=True)
                resource_estimate(model, model_bytes=descriptor_bytes,
                    extra_bytes=self.retained_bytes+self.guard_scratch_bytes+34*_owned(manifest)+65536)
                options = {'version': '3', 'checkpoint': str(directory), 'expected_manifest_digest': sha,
                    'model': asdict(model.architecture()), 'inference': asdict(model.inference),
                    'resources': asdict(model.resources), 'action_config_hash': digest(asdict(actions)),
                    'tree_descriptor': clone(manifest['tree_descriptor']), 'feature_signature': manifest['feature_signature'],
                    'source_kind': manifest['source_kind'], 'source_origin': clone(manifest['source_origin']),
                    'training_pair': {'request_sha256': binding.request_sha256, 'completion_sha256': binding.completion_sha256},
                    **({'storage_binding': clone(manifest['storage_binding'])} if 'storage_binding' in manifest else {})}
                entry = {'input_contract': TREE_INPUT_CONTRACT, 'policy_options': options,
                    'inference_manifest': clone(manifest), 'files': [
                        {'path': name, 'sha256': pair.scan.files[directory / name][1],
                         'identity': list(pair.scan.files[directory / name][0])}
                        for name in ('inference.json', 'model.txt')]}
                _, label = validate_tree_metadata(entry, actions=actions, rules=rules)
                pair.scan.charge(34 * len(canonical(entry).encode()) + 65536)
                pair.finalize()
                if label in self._entries and canonical(self._entries[label]) != canonical(entry):
                    raise ContractError('one Tree policy label cannot name distinct originals')
                self._entries[label] = entry
                self._labels[digest(asdict(model))] = label
        except BaseException:
            self._failed = True
            self.release()
            raise

    def _live(self) -> None:
        if self._released or self._failed:
            raise ContractError('retained game policy originals are released or failed')

    @property
    def retained_bytes(self) -> int:
        self._live()
        neural = 0 if self._neural is None else self._neural.retained_bytes
        return neural + sum(pair.scan.retained for pair in self._pairs)

    @property
    def guard_scratch_bytes(self) -> int:
        self._live()
        # Snapshot list stays live during hashing and the largest inventory
        # sweep. Hash buffers are bounded at1MiB by the shared SealedScan.
        tree = max((1024**2 + 512 + 128*len(pair.scan.files) + max(
            (65536 + 2048*len(names) + 8*sum(len(name.encode()) for name in names)
             for names in pair.scan.inventories.values()), default=65536)
            for pair in self._pairs), default=0)
        # The public NN counter includes >=512 bytes per sealed path, charged
        # inventory names and32x JSON/record ownership. Eight times that ACTUAL
        # retained charge bounds concurrent snapshot/inventory + finalize-copy
        # storage without inspecting private NN paths or reserving its full cap.
        neural = (0 if self._neural is None else
            1024**2 + 65536 + 8*self._neural.retained_bytes)
        return max(tree, neural)

    def guard(self, *, deadline: float) -> None:
        import math
        self._live()
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ContractError('finite original game artifact guard deadline required')
        try:
            if self.retained_bytes + self.guard_scratch_bytes > self._limit:
                raise ContractError('joint retained game artifact guard scratch admission exceeded')
            for pair in self._pairs:
                pair.scan.deadline = min(deadline, self._deadline)
                pair.finalize()
            if self._neural is not None:
                # Existing public finalization checks all original NN snapshots
                # and its original deadline. Its safe metadata clone is transient.
                self._neural.finalize(deadline=min(deadline, self._deadline))
        except BaseException:
            self._failed = True
            raise

    def binding(self, model: Any) -> tuple[str, Any]:
        from gf_common.domain import RuntimePolicyBinding
        self._live()
        if type(model) is ConfiguredTreeModelV3:
            label = self._labels.get(digest(asdict(model)))
            if label is None:
                raise ContractError('Tree Config is not an owned original')
            return label, RuntimePolicyBinding('lightgbm', clone(self._entries[label]['policy_options']))
        if self._neural is None:
            raise ContractError('no retained neural originals')
        return self._neural.binding(model)

    def require_options(self, model_id: str, options: Json) -> None:
        self._live()
        if model_id == 'lightgbm':
            if not any(canonical(entry['policy_options']) == canonical(options) for entry in self._entries.values()):
                raise ContractError('Tree worker differs from retained actual pair')
        elif self._neural is not None:
            self._neural.require_options(model_id, options)
        else:
            raise ContractError('unknown retained own policy')

    def journal_metadata(self, policy_label: str) -> Json:
        self._live()
        if policy_label in self._entries:
            from gf_models.structured_tree_config import resource_estimate
            from gf_models.structured_tree_encoder import _owned
            entry = self._entries[policy_label]
            resource_estimate(_configured(entry['policy_options']),
                model_bytes=entry['policy_options']['tree_descriptor']['model_bytes'],
                extra_bytes=self.retained_bytes+self.guard_scratch_bytes+34*_owned(entry)+65536)
            return clone(entry)
        if self._neural is not None:
            return self._neural.journal_metadata(policy_label)
        raise ContractError('unknown retained policy label')

    def finalize_inputs(self, *, deadline: float) -> Json:
        self.guard(deadline=deadline)
        result: Json = {}
        if self._neural is not None:
            result['neural_inputs'] = self._neural.finalize(deadline=deadline)
        if self._entries:
            result['tree_inputs'] = {'input_contract': TREE_INPUT_CONTRACT, 'policies': clone(self._entries)}
        return result

    def retain_worker(self, worker: Any) -> None:
        self._live()
        self._workers.append(worker)

    def retire_worker(self, worker: Any) -> None:
        # Cleanup remains available after an admitted artifact failure. Removal
        # only follows confirmed physical retirement, never a closed flag.
        if not worker.lease.released or worker.io_thread is not None and worker.io_thread.is_alive():
            raise ContractError('game worker must retire lease and I/O before artifact deregistration')
        if worker in self._workers:
            self._workers.remove(worker)

    @property
    def can_release(self) -> bool:
        return not any(not worker.lease.released or worker.io_thread is not None and worker.io_thread.is_alive()
                       for worker in self._workers)

    def release(self) -> None:
        if not self.can_release:
            raise ContractError('actual Tree3 pair remains retained until every worker and I/O thread retires')
        # Caller retains this owner until all processes and I/O threads retire.
        # After publication this is deliberately memory-only.
        self._released = True
        if self._neural is not None:
            self._neural.release()
        self._entries.clear()
        self._labels.clear()
        for pair in self._pairs:
            pair.scan.files.clear()
            pair.scan.inventories.clear()
        self._pairs.clear()
        self._workers.clear()


class _OwnGameTreePolicyV3:
    """Private child execution handle: owns only its own files and atomic policy inputs."""
    def __init__(self, options: Json, *, actions: ActionConfigV3, rules: CardInstanceRules,
                 managed_limit: int, deadline: float) -> None:
        import time
        from pathlib import Path

        from gf_models.structured_sealed import SealedScan
        from gf_models.structured_tree_artifacts import _load_game_own_model_v3
        from gf_models.structured_tree_config import resource_estimate
        from gf_models.structured_tree_encoder import _owned
        safe = safe_tree_options_game_v3(options, actions=actions, rules=rules)
        model = _configured(safe)
        if model.resources.max_owned_bytes != managed_limit:
            raise ContractError('Tree3 sanitized argv and declared managed cap differ')
        self._scan = SealedScan(max_owned_bytes=managed_limit, deadline=deadline)
        directory = Path(safe['checkpoint']).absolute()
        self._scan.inventory(directory, {'inference.json', 'model.txt'})
        manifest = self._scan.object(directory / 'inference.json', safe['expected_manifest_digest'])
        files = [
            {'path': 'inference.json', 'sha256': safe['expected_manifest_digest'],
             'identity': list(self._scan.files[directory / 'inference.json'][0])}]
        descriptor = TreeDescriptor(**safe['tree_descriptor'])
        with self._scan.file(directory / 'model.txt', descriptor.model_sha256) as (stream, info):
            if info.st_size != descriptor.model_bytes:
                raise ContractError('Tree3 own native model length differs')
            metadata_bytes = self._scan.files[directory / 'inference.json'][0][2]
            resource_estimate(model, model_bytes=info.st_size,
                extra_bytes=self._scan.retained + _owned(safe) + 34*(metadata_bytes+info.st_size) + 65536)
            text = stream.read(info.st_size+1)
            if len(text) != info.st_size:
                raise ContractError('Tree3 own native model changed during read')
            files.append({'path': 'model.txt', 'sha256': descriptor.model_sha256,
                'identity': list(self._scan.files[directory / 'model.txt'][0])})
        entry = {'input_contract': TREE_INPUT_CONTRACT, 'policy_options': safe,
            'inference_manifest': manifest, 'files': files}
        _, self.version = validate_tree_metadata(entry, actions=actions, rules=rules)
        with self._scan.file(directory / 'inference.json', safe['expected_manifest_digest']) as (stream, info):
            metadata = stream.read(info.st_size+1)
            if len(metadata) != info.st_size:
                raise ContractError('Tree3 own metadata changed during read')
        self.tree = _load_game_own_model_v3(metadata, text,
            expected_digest=safe['expected_manifest_digest'], model=model,
            action_config_hash=safe['action_config_hash'], retained_bytes=self._scan.retained+_owned((safe, metadata, text)),
            deadline=deadline, own_pair_reference=safe['training_pair'])
        self._scan.finalize()
        self.model, self._actions, self._rules = model, actions, rules
        self.last_timing: dict[str, float] = {}
        self.reserved_bytes = resource_estimate(model, model_bytes=descriptor.model_bytes)['total_bytes']
        if time.monotonic() >= deadline:
            raise TimeoutError('Tree3 own model load deadline exhausted')

    def reset(self, sampling_seed: int) -> None:
        integer(sampling_seed)
        self.last_timing = {}

    def retained_bytes(self, extra: Any = None, extra_bytes: int = 0) -> int:
        from gf_models.structured_tree_config import resource_estimate
        from gf_models.structured_tree_encoder import _owned
        return resource_estimate(self.model, model_bytes=self.tree.descriptor.model_bytes,
            extra_bytes=self._scan.retained+_owned(extra)+extra_bytes)['total_bytes']

    def select_frame(self, frame: Any, bank_view: Any, generated: Any) -> Any:
        import time

        from gf_common.diverse_actions import GameGeneratedCandidatesV3
        from gf_common.structured import Selection
        from gf_models.structured_tree_artifacts import _bind_model_v3
        from gf_models.structured_tree_pairing import game_records
        from gf_models.structured_tree_policy import _prepare_policy, _select
        if type(generated) is not GameGeneratedCandidatesV3:
            raise ContractError('Tree3 game policy requires the original game-generated table')
        table = generated.table
        if (table.purpose != 'qualified_game' or table.synthetic_capacity
                or table.action_contract != self._actions.action_contract
                or generated.report.config_hash != digest(asdict(self._actions))):
            raise ContractError('Tree3 original game candidate binding mismatch')
        table.validate(frame)
        self._scan.deadline = time.monotonic()+10
        self._scan.finalize()
        _bind_model_v3(self.tree.booster, self.tree.model, self.tree.origin)
        prepared, candidates, token_ms = _prepare_policy(self.tree, frame, bank_view, table, game_records(self.tree.origin.fields()))
        index = _select(self.tree, prepared, candidates)
        self._scan.finalize()
        self.last_timing = {'tokenization_ms': token_ms}
        return Selection(table.table_hash, index, self.version, None, None)
