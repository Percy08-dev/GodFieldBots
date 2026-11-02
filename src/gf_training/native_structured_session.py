"""Concrete structured native Source owner/authorization; no formal eligibility.

Native owns originals and raw-input staging; Python owns sealed NPZ validation,
PCG/schedule reconstruction, full-precision GAE and C1 publication/inspection.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import numpy as np

from gf_common.action_contract import parse_action_contract
from gf_common.domain import ContractError, Json, RuntimePolicyBinding, canonical, closed, digest
from gf_common.schedule import serialize_schedule
from gf_models.structured_binding import recorded_policy_label
from gf_models.structured_config import InferenceBindingV2, recorded_model_config_v2
from gf_models.structured_sealed import (
    SealedScan,
    inspect_pruned_inference,
    inspect_retained_artifact,
)
from gf_training.collection import training_groups


def _authorize_controls(summary: Json, native: Any, *, max_owned_bytes: int, deadline: float,
                        extra_owned_bytes: int = 0) -> tuple[Json, SealedScan, dict[int, Any]]:
    """Actual NumPy-only sealed checks and existing schedule algorithm, no flags.

    The native session is paused and retains its original control/Python-summary
    leases. S2's one cumulative budget is only the REMAINING caller allowance.
    Keep the returned scan alive and charged until final source rechecks.
    """
    closed(summary, {'schema_version', 'kind', 'reader_contract', 'scope', 'mode', 'qualified',
        'parent_qualification', 'action_contract', 'purpose', 'synthetic_capacity', 'control_fingerprint',
        'authorization_max_bytes', 'parents', 'collections'})
    if (type(summary['schema_version']) is not int or summary['schema_version'] != 2
            or summary['kind'] != 'structured-source-controls' or summary['reader_contract'] != 'structured-v1'
            or summary['scope'] != 'engineering-component-originals' or summary['qualified'] is not False
            or summary['parent_qualification'] != 'not_performed' or summary['purpose'] != 'component'
            or summary['synthetic_capacity'] is not False):
        raise ContractError('closed engineering original Source summary2 required')
    contract = parse_action_contract(summary['action_contract'])
    remaining = max_owned_bytes - native.used_bytes() - extra_owned_bytes
    if remaining <= 0:
        raise ContractError('joint native/sealed ownership exhausted before inspection')
    scan = SealedScan(max_owned_bytes=remaining, deadline=deadline)
    parents, models, schedules, registries = [], {}, {}, {}
    for pi, parent in enumerate(summary['parents']):
        scan.check()
        if parent['parent_index'] != pi:
            raise ContractError('ordered original parent index mismatch')
        model = recorded_model_config_v2(parent['model'], parent['semantics']['token_signature'])
        models[pi] = model
        generations = parent['generations']
        binding = InferenceBindingV2(**parent['execution']['inference']['binding'])
        visibility = parent['execution']['inference']['cuda_visible_devices']
        def policy(number: int, generations: Any = generations, model: Any = model,
                   binding: InferenceBindingV2 = binding, visibility: Any = visibility,
                   parent: Json = parent) -> tuple[str, RuntimePolicyBinding]:
            record = generations[number]
            label = recorded_policy_label(model, binding, record['inference_sha256'], cuda_visible_devices=visibility)
            return label, RuntimePolicyBinding(model.model_id, {'version': '2',
                'checkpoint': str(Path(parent['directory']) / 'artifacts' / f'{number:06}'),
                'expected_manifest_digest': record['inference_sha256'], 'model': asdict(model),
                'inference': asdict(binding), 'cuda_visible_devices': visibility,
                'action_contract': asdict(contract), 'purpose': 'component', 'synthetic_capacity': False})
        initial = np.random.Generator(np.random.PCG64(parent['state']['seed']))
        initial_controls = {'schedule_opponent_rng': dict(initial.bit_generator.state), 'snapshot_pool': [0], 'collection_position': 0}
        if canonical(generations[0]['parent_controls']) != canonical(initial_controls):
            raise ContractError('original generation0 PCG64 differs from frozen seed')
        sealed = []
        for gi, generation in enumerate(generations):
            scan.check()
            if generation['generation'] != gi:
                raise ContractError('ordered original generation index mismatch')
            directory = Path(parent['directory']) / 'artifacts' / f'{gi:06}'
            if generation['artifact_status'] == 'retained':
                actual = inspect_retained_artifact(directory, expected_inference_digest=generation['inference_sha256'],
                    action_contract=contract, purpose='component', synthetic_capacity=False, scan=scan)
                if actual.resume is None or scan.files[directory / 'resume.json'][1] != generation['resume_sha256']:
                    raise ContractError('sealed original resume link mismatch')
                if canonical(actual.resume['binding']) != canonical(parent['execution']['learner']['binding']):
                    raise ContractError('sealed original learner execution differs')
            else:
                descriptor = generation['pruned_descriptor']
                actual = inspect_pruned_inference(directory,
                    expected_inference_digest=generation['inference_sha256'], expected_resume_digest=generation['resume_sha256'],
                    original_files=descriptor['original_files'], deletion_receipt={
                        'status': 'deleted', 'inference_sha256': generation['inference_sha256'],
                        'original_files': descriptor['original_files'],
                        'deletion_intent': descriptor['deletion_intent'], 'deletion_receipt': descriptor['deletion_receipt']},
                    action_contract=contract, purpose='component', synthetic_capacity=False, scan=scan)
            if canonical(actual.inference['model']) != canonical(asdict(model)) or digest(actual.files) != generation['artifact_files_sha256']:
                raise ContractError('sealed original full model/resources/file map differs')
            sealed.append({'generation': gi, 'inference_sha256': generation['inference_sha256'],
                'resume_sha256': generation['resume_sha256'], 'artifact_files_sha256': digest(actual.files),
                'retained': actual.retained})
            # Reconstruct all original train schedules with the unchanged helper.
            label, learner = policy(gi)
            registry = {'random-structured-v1': RuntimePolicyBinding('random'),
                        'heuristic-structured-v1': RuntimePolicyBinding('heuristic'), label: learner}
            snapshots = []
            controls = generation['parent_controls']
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = controls['schedule_opponent_rng']
            for number in controls['snapshot_pool']:
                name, value = policy(number)
                registry[name] = value
                snapshots.append(name)
            groups = training_groups(parent['state']['seed'], gi, label, tuple(snapshots), rng, actions_version=2)
            schedules[pi, gi] = digest(serialize_schedule(groups, 4))
            registries[pi, gi] = digest({name: asdict(value) for name, value in registry.items()})
            if gi + 1 < len(generations):
                expected = {'schedule_opponent_rng': dict(rng.bit_generator.state),
                    'snapshot_pool': [number for number in range(gi + 2) if number % 10 == 0][-5:],
                    'collection_position': gi + 1}
                if canonical(generations[gi + 1]['parent_controls']) != canonical(expected):
                    raise ContractError('original schedule/pool/PCG advance differs')
        parents.append({'parent_index': pi, 'config_sha256': parent['state']['config_sha256'],
            'resume_identity': parent['state']['resume_identity'], 'sealed': sealed,
            'controls_sha256': digest([generation['parent_controls'] for generation in generations])})
    collections = []
    for ci, collection in enumerate(summary['collections']):
        scan.check()
        if collection['collection_index'] != ci:
            raise ContractError('ordered original collection index mismatch')
        parent, generation = collection['parent_index'], collection['generation']
        if parent is None:
            if collection['schema_version'] != 3 or collection['correspondence'] is not None or collection['numeric'] is not None:
                raise ContractError('raw collection cannot invent neural/PPO ancestry')
            schedule, bindings = collection['schedule_sha256'], collection['bindings_sha256']
            correspondence, numeric = None, None
        else:
            schedule, bindings = schedules[parent, generation], registries[parent, generation]
            if schedule != collection['schedule_sha256'] or bindings != collection['bindings_sha256']:
                raise ContractError('original native source differs from frozen schedule/bindings')
            correspondence = collection['correspondence']['binding_sha256']
            numeric = collection['numeric']['sha256'] if collection['numeric'] is not None else None
        collections.append({'collection_index': ci, 'manifest_sha256': collection['manifest_sha256'],
            'expected_schedule_sha256': schedule, 'expected_bindings_sha256': bindings,
            'source_set_sha256': collection['source_set_sha256'], 'correspondence_sha256': correspondence,
            'numeric_sha256': numeric})
    scan.finalize()
    if time.monotonic() >= deadline:
        raise TimeoutError('structured Source authorization deadline')
    return {'schema_version': 2, 'mode': summary['mode'], 'control_fingerprint': summary['control_fingerprint'],
            'parents': parents, 'collections': collections}, scan, models


class NativeStructuredSource:
    """One concrete Source session; no public real-I/O/qualified-source bypass.

    Runtime entrypoints separately require the accepted native capability. An
    explicitly supplied candidate module is for private component composition.
    """
    def __init__(self, plan: Json, settings: Any, *, max_owned_bytes: int, deadline: float,
                 extra_owned_bytes: int = 0, max_scratch_bytes: int | None = None, native: Any = None) -> None:
        from contextlib import ExitStack

        from gf_common.preprocessing import preparation_config
        from gf_training.structured_rollout import _owned_bytes
        self.settings = preparation_config(asdict(settings))
        if self.settings.backend != 'rust' or self.settings.reader != 'structured-v1':
            raise ContractError('explicit structured native Source consumer required')
        self.plan = plan
        self.limit = min(max_owned_bytes, self.settings.memory_budget_mib * 1024**2)
        self.deadline = min(deadline, time.monotonic() + self.settings.wall_seconds - self.settings.cleanup_seconds)
        self.scratch_limit = self.settings.scratch_budget_mib * 1024**2
        if max_scratch_bytes is not None:
            self.scratch_limit = min(self.scratch_limit, max_scratch_bytes)
        self.extra, self.native = extra_owned_bytes, native
        self.base = 65536 + _owned_bytes((plan, self.settings))
        self.budget = self.limit - self.extra - self.base
        if self.budget <= 0 or self.scratch_limit <= 0:
            raise ContractError('native Source caller ownership/scratch exhausted before I/O')
        self._locks = ExitStack()
        self._executor: Any = None
        self._session: Any = None
        self.summary: Json | None = None
        self.sealed: SealedScan | None = None
        self.models: dict[int, Any] = {}
        self._auth: Json | None = None
        self._closed = False
        self._session_closed = False
        self._correspondence_done = False
        self._begun = False
        self._published: Json | None = None
        self._persistent_python = 0

    def _check(self) -> None:
        if self._closed:
            raise ContractError('structured Source owner is closed')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('structured Source owner deadline')

    def _held(self, additional: int = 0) -> int:
        return (self.sealed.retained if self.sealed is not None else 0) + self._persistent_python + additional

    def __enter__(self) -> Self:
        from gf_training.lineage import _OWNED_LOCKS, run_lock
        from gf_training.native_backend import binary_identity, load_native
        from gf_training.native_execution import NativeExecution
        from gf_training.structured_rollout import _owned_bytes
        try:
            self._check()
            self.native = self.native or load_native()
            if not callable(getattr(self.native, 'StructuredNativeSession', None)):
                raise ContractError('native binary lacks the concrete structured Source session')
            self.identity = binary_identity(self.native)
            self.base += _owned_bytes(self.identity)
            self.budget = self.limit - self.extra - self.base
            if self.budget <= 0:
                raise ContractError('native Source binary/control ownership exhausted before I/O')
            for entry in sorted(self.plan['parents'], key=lambda value: value['directory']):
                directory = Path(entry['directory']).resolve()
                if directory not in _OWNED_LOCKS:
                    self._locks.enter_context(run_lock(directory))
            settings = canonical({'memory_bytes': self.budget, 'scratch_bytes': self.scratch_limit,
                'max_source_file_bytes': self.settings.max_source_file_mib * 1024**2,
                'max_record_bytes': self.settings.max_record_mib * 1024**2,
                'chunk_target_bytes': min(self.settings.chunk_target_mib * 1024**2, 64 * 1024**2)}).encode()
            encoded_plan = canonical(self.plan).encode()
            cancel = self.native.CancelHandle(self.deadline)
            self._executor = NativeExecution(cancel, thread_name_prefix='gf-structured-source')
            def construct() -> None:
                self._session = self.native.StructuredNativeSession(encoded_plan, settings, self.deadline, cancel)
            self._executor.call(construct)
            encoded = self._executor.call(self._session.control_summary)
            self.summary = __import__('json').loads(encoded)
            self._auth, self.sealed, self.models = _authorize_controls(self.summary, self._session,
                max_owned_bytes=self.budget, deadline=self.deadline)
            if self.plan['mode'] == 'historical_parent':
                self._check_historical()
                self._correspondence_done = True
            self._executor.call(self._session.authorize, canonical(self._auth).encode(), self._held())
            self._check()
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - initiating error wins owned cleanup
                error.add_note('structured Source cleanup failed: ' + type(cleanup).__name__)
                if getattr(cleanup, '_structured_source_cleanup_owner', None) is self:
                    vars(error)['_structured_source_cleanup_owner'] = self
            raise

    def _check_historical(self) -> None:
        from gf_common.action_contract import parse_action_contract
        from gf_training.structured_rollout import _read_recorded_rollout
        assert self.summary is not None
        contract = parse_action_contract(self.summary['action_contract'])
        for collection in self.summary['collections']:
            self._check()
            numeric = collection['numeric']
            if numeric is None:
                continue
            self.recheck()
            rollout = _read_recorded_rollout(Path(numeric['directory']), numeric['sha256'],
                self.models[collection['parent_index']], action_contract=contract,
                declaration=collection['semantics'], expected_records=None,
                expected_correspondence=collection['correspondence'], max_owned_bytes=self.budget,
                extra_owned_bytes=self._session.used_bytes(), check=self._check,
                reserve=lambda size: self._reserve_decode(size))
            rollout.require_learning()
            if len(rollout) != collection['learner_decisions']:
                raise ContractError('historical learner correspondence count mismatch')
            del rollout
            self.recheck()

    def _reserve_decode(self, size: int) -> None:
        self._check()
        if size + self._session.used_bytes() > self.budget:
            raise ContractError('joint native/recorded decoder ownership exhausted')

    def recheck(self, additional: int = 0) -> None:
        self._check()
        if self.sealed is not None:
            self.sealed.finalize()
        self._executor.call(self._session.recheck, self._held(additional))

    def begin_update(self) -> None:
        self._check()
        if self.plan['mode'] != 'live_cohort' or not self._correspondence_done or self._begun:
            raise ContractError('actual live original correspondence must precede one update phase')
        self.recheck()  # Both ORIGINAL native and sealed owners, immediately before Adam.
        self._executor.call(self._session.begin_update, self._held())
        self._begun = True

    def finalize(self) -> Json:
        self._check()
        if not self._correspondence_done or self.plan['mode'] == 'live_cohort' and not self._begun:
            raise ContractError('actual Source comparison/update phase required before finalization')
        self.recheck()
        value = __import__('json').loads(self._executor.call(self._session.finalize, self._held()))
        if self.sealed is not None:
            self.sealed.finalize()
        return value

    def close(self) -> None:
        if self._closed:
            return
        error = None
        if self._executor is not None and self._session is not None and not self._session_closed:
            try:
                self._executor.call(self._session.close)
                self._session_closed = True
            except BaseException as failure:  # noqa: BLE001 - retry confirms retirement vs retained cleanup
                error = failure
                try:
                    self._executor.call(self._session.close)
                    self._session_closed = True
                except BaseException as cleanup:  # noqa: BLE001 - retain owner/locks if cleanup stays unconfirmed
                    error.add_note('structured Source still owns cleanup: ' + type(cleanup).__name__)
                    vars(error)['_structured_source_cleanup_owner'] = self
                    raise error
        if self._executor is not None:
            try:
                self._executor.close()
            except BaseException as shutdown:  # noqa: BLE001 - shutdown may join before preserving cancellation
                if error is None:
                    error = shutdown
                else:
                    error.add_note('structured Source executor shutdown: ' + type(shutdown).__name__)
                if not self._executor._closed or self._executor._active is not None:
                    vars(error)['_structured_source_cleanup_owner'] = self
                    raise error
        self._locks.close()
        self._closed = True
        if error is not None:
            raise error

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        failure = None
        if value is None:
            try:
                self.finalize()
            except BaseException as error:  # noqa: BLE001 - clean up before propagating final failure
                failure = error
        try:
            self.close()
        except BaseException as error:
            original = value if isinstance(value, BaseException) else failure
            if original is None:
                raise
            original.add_note('structured Source cleanup failed: ' + type(error).__name__)
            if getattr(error, '_structured_source_cleanup_owner', None) is self:
                vars(original)['_structured_source_cleanup_owner'] = self
        if failure is not None:
            raise failure


    def write_numeric(self, directory: Path) -> tuple[int, Json, Json]:
        """One export/publication/retirement/comparison cycle for the selected cohort."""
        from gf_models.arrays import sha256
        from gf_training.structured_rollout import (
            _owned_bytes,
            _read_recorded_rollout,
            _write_prepared_payload,
        )
        self._check()
        if self.plan['mode'] != 'live_cohort' or self._published is not None:
            raise ContractError('one original live neural cohort publication required')
        assert self.summary is not None
        index = self.plan['cohort_collection_index']
        collection = self.summary['collections'][index]
        model = self.models[collection['parent_index']]
        payload = self._executor.call(self._session.export_cohort, self._held())
        metadata, arrays = _payload_views(payload)
        if metadata['control_fingerprint'] != self.summary['control_fingerprint'] or metadata['collection_index'] != index:
            raise ContractError('exported cohort differs from authorized original Source')
        if canonical(metadata['correspondence']) != canonical(collection['correspondence']):
            raise ContractError('exported original correspondence differs from authorized Source')
        # Label buffers are additional Python ownership, not native learned latents.
        self._reserve_decode(65536 + 128 * metadata['metadata']['decision_count'])
        labels = _attach_gae(metadata['metadata'], arrays)
        manifest = _write_prepared_payload(directory, metadata['metadata'], arrays, model,
            resident_bytes=_owned_bytes((model, labels)), extra_owned_bytes=self._session.used_bytes(),
            max_owned_bytes=self.budget, shard_bytes=min(self.settings.chunk_target_mib * 1024**2, 64 * 1024**2),
            check=self._check, reserve=self._reserve_decode)
        proof = metadata['source_proof']
        expected = metadata['correspondence']
        self._published = manifest
        self._persistent_python = _owned_bytes((manifest, proof, expected))
        del labels, arrays, metadata, payload
        self._executor.call(self._session.retire_payload, self._held())
        numeric_hash = sha256(directory / 'rollout.json', self.deadline)
        result = _read_recorded_rollout(directory, numeric_hash, model, action_contract=parse_action_contract(self.summary['action_contract']),
            declaration=collection['semantics'], expected_records=None, expected_correspondence=expected,
            max_owned_bytes=self.budget, extra_owned_bytes=self._session.used_bytes(), check=self._check, reserve=self._reserve_decode)
        result.require_learning()
        if result.policy_version != collection['learner_label'] or len(result) != collection['learner_decisions']:
            raise ContractError('published numeric original policy/endpoint count differs')
        count = len(result)
        del result
        self.recheck()
        self._correspondence_done = True
        return count, proof, {'rollout.json': numeric_hash, **manifest['data']['files']}

def _payload_views(payload: Any) -> tuple[Json, dict[str, Any]]:
    """Validate closed bank descriptors before creating any named NumPy view."""
    import json
    import math

    from gf_common.domain import integer
    if type(payload) is not tuple or len(payload) != 4 or any(type(banks) is not tuple for banks in payload[1:]):
        raise ContractError('closed typed-bank tuple payload required')
    metadata = json.loads(payload[0])
    closed(metadata, {'schema_version', 'kind', 'control_fingerprint', 'collection_index', 'metadata',
                      'arrays', 'source_proof', 'correspondence'})
    if type(metadata['schema_version']) is not int or metadata['schema_version'] != 1 or metadata['kind'] != 'structured-cohort-payload':
        raise ContractError('unsupported structured cohort payload')
    banks = {'uint8': payload[1], 'int64': payload[2], 'float32': payload[3]}
    for dtype, values in banks.items():
        for value in values:
            if (type(value) is not np.ndarray or value.dtype != np.dtype(dtype) or value.ndim != 1
                    or value.flags.writeable or not value.flags.c_contiguous or value.nbytes > 64 * 1024**2):
                raise ContractError('typed bank dtype/readonly/size mismatch')
    ranges: dict[tuple[str, int], list[tuple[int, int]]] = {}
    names = set()
    plans = []
    if type(metadata['arrays']) is not list:
        raise ContractError('ordered typed array descriptors required')
    for item in metadata['arrays']:
        closed(item, {'key', 'dtype', 'shape', 'bank_index', 'element_offset', 'element_count'})
        key, dtype = item['key'], item['dtype']
        if (type(key) is not str or not key or not key.isascii()
                or not all(character.isalnum() or character in '._-' for character in key)
                or key in names or dtype not in banks):
            raise ContractError('typed array key/dtype ownership mismatch')
        names.add(key)
        index = integer(item['bank_index'], 0, len(banks[dtype]) - 1)
        source = banks[dtype][index]
        offset = integer(item['element_offset'], 0, len(source))
        count = integer(item['element_count'], 0, len(source) - offset)
        shape = item['shape']
        if type(shape) is not list or len(shape) > 8:
            raise ContractError('typed shape required')
        for axis in shape:
            integer(axis, 0, 64 * 1024**2)
        if math.prod(shape) != count or count * source.dtype.itemsize > 64 * 1024**2:
            raise ContractError('typed descriptor scalar count/size mismatch')
        ranges.setdefault((dtype, index), []).append((offset, offset + count))
        plans.append((key, source, offset, count, tuple(shape)))
    for spans in ranges.values():
        spans.sort()
        if any(left[1] > right[0] for left, right in pairwise(spans)):
            raise ContractError('overlapping typed array owners')
    return metadata, {key: source[offset:offset + count].reshape(shape) for key, source, offset, count, shape in plans}


def _attach_gae(metadata: Json, arrays: dict[str, Any]) -> list[Any]:
    """Use FULL JSON Selection values and normalize ONE original collection."""
    from gf_training.ppo import gae, normalize_advantages
    raw = []
    computed: list[tuple[Any, Any] | None] = []
    for episode in metadata['episodes']:
        values = [selection['value'] for selection in episode['selections']]
        targets = episode['end_kind'] == 'natural' and bool(values)
        episode['has_targets'] = targets
        if targets:
            outcome = episode['outcome']
            reward = 0 if outcome == 'draw' else 1 if outcome['win']['winner'] == episode['actor'] else -1
            advantages, target = gae(values, reward)
            raw.extend(advantages.tolist())
            computed.append((advantages, target.astype(np.float32)))
        else:
            computed.append(None)
    normalized = normalize_advantages(raw) if raw else np.empty(0, np.float32)
    offset = 0
    owned = [normalized]
    for index, entry in enumerate(computed):
        if entry is None:
            continue
        count = len(entry[0])
        advantage = normalized[offset:offset + count].copy()
        offset += count
        target = entry[1]
        advantage.flags.writeable = target.flags.writeable = False
        arrays[f'e.{index}.advantage'] = advantage
        arrays[f'e.{index}.target'] = target
        owned.extend((advantage, target))
    return owned



def _frozen_native_plan(parent: Json, generation: int, deadline: float) -> tuple[Any, str]:
    """Existing schedule helper over compact ORIGINAL parent metadata only."""
    from gf_training.collection import CollectionPlan
    model = recorded_model_config_v2(parent['model'], parent['semantics']['token_signature'])
    binding = InferenceBindingV2(**parent['execution']['inference']['binding'])
    visibility = parent['execution']['inference']['cuda_visible_devices']
    contract = parse_action_contract(parent['action_contract'])
    registry = {'random-structured-v1': RuntimePolicyBinding('random'), 'heuristic-structured-v1': RuntimePolicyBinding('heuristic')}
    def policy(number: int) -> tuple[str, RuntimePolicyBinding]:
        record = parent['generations'][number]
        label = recorded_policy_label(model, binding, record['inference_sha256'], cuda_visible_devices=visibility)
        return label, RuntimePolicyBinding(model.model_id, {'version': '2',
            'checkpoint': str(Path(parent['directory']) / 'artifacts' / f'{number:06}'),
            'expected_manifest_digest': record['inference_sha256'], 'model': asdict(model), 'inference': asdict(binding),
            'cuda_visible_devices': visibility, 'action_contract': asdict(contract), 'purpose': 'component', 'synthetic_capacity': False})
    label, learner = policy(generation)
    registry[label] = learner
    controls = parent['generations'][generation]['parent_controls']
    snapshots = []
    for number in controls['snapshot_pool']:
        name, value = policy(number)
        registry[name] = value
        snapshots.append(name)
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = controls['schedule_opponent_rng']
    groups = training_groups(parent['state']['seed'], generation, label, tuple(snapshots), rng, actions_version=2)
    batch = f'batch-{generation:06}'
    linkage = {'parent_run_id': parent['state']['run_id'], 'parent_run_path': parent['directory'], 'batch_id': batch,
        'collection_generation': generation, 'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes}}
    return CollectionPlan(groups, registry, Path(parent['directory']) / 'collections' / batch, linkage, deadline, deadline), label


def _validate_owned_metadata(plan: Any, ledger: Any, parent: Json, *, deadline: float) -> None:
    """Preserve concrete owner/lock/ledger/identity checks without Python Source."""
    from gf_models.structured_config import ModelConfigV2
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.lineage import _OWNED_LOCKS
    from gf_training.structured_learning_process import StructuredLearningProcess
    from gf_training.structured_numeric_worker import wire_identity
    if not isinstance(ledger, ReservationLedger) or ledger.learner is None or ledger.updating or ledger.workers:
        raise ContractError('owned collection requires the SAME live learner ledger before participant startup')
    lease = ledger.learner
    owner = lease._cleanup_owner
    if (not isinstance(owner, StructuredLearningProcess) or owner.ledger is not ledger or owner.lease is not lease
            or lease.released or lease.creation_pending or owner.steps is None or owner.process is None
            or owner.process.returncode is not None or lease.pid != owner.process.pid):
        raise ContractError('owned collection concrete learner owner is missing/mismatched')
    directory = Path(plan.linkage['parent_run_path']).resolve()
    if directory not in _OWNED_LOCKS or owner.directory.resolve() != directory or directory != Path(parent['directory']):
        raise ContractError('owned collection requires original parent lock/learner')
    model = recorded_model_config_v2(parent['model'], parent['semantics']['token_signature'])
    if type(model) is not ModelConfigV2:
        raise ContractError('historical descriptor cannot authorize active participants')
    limits = parent['run_limits']
    if (ledger.global_cuda_bytes != limits['global_cuda_budget_bytes'] or ledger.max_workers != limits['max_participant_workers']
            or ledger.learner_bytes != 0):
        raise ContractError('owned collection ledger limits differ from frozen parent')
    execution = parent['execution']['learner']
    expected_identity = wire_identity(model, InferenceBindingV2(**execution['binding']), parent['actions_sha256'],
        parent['state']['resume_identity'], action_contract=parse_action_contract(parent['action_contract']),
        purpose='component', synthetic_capacity=False, cuda_visible_devices=execution['cuda_visible_devices'])
    expected, _ = _frozen_native_plan(parent, parent['state']['updates'], deadline)
    if (parent['state']['status'] != 'running' or canonical(plan.linkage) != canonical(expected.linkage)
            or canonical(serialize_schedule(plan.groups, 4)) != canonical(serialize_schedule(expected.groups, 4))
            or canonical({key: asdict(value) for key, value in plan.bindings.items()}) != canonical({key: asdict(value) for key, value in expected.bindings.items()})
            or plan.output.resolve() != expected.output or plan.kind != 'ppo_batch'
            or canonical(asdict(owner.model)) != canonical(asdict(model))
            or canonical(asdict(owner.binding)) != canonical(execution['binding'])
            or canonical(owner.identity) != canonical(expected_identity)
            or owner.visibility != execution['cuda_visible_devices'] or owner.steps != parent['state']['optimizer_steps']
            or canonical(owner.hardware) != canonical(parent['hardware'])):
        raise ContractError('owned collection differs from original frozen plan/concrete learner identity')
