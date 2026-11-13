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


def _verify_parent_controls(original_parents: list[Json], contract: Any, scan: SealedScan, *, game_actions: Any = None) -> tuple[list[Json], dict[int, Any], dict[tuple[int, int], str], dict[tuple[int, int], str]]:
    """Shared recorded-only seals/PCG/cadence checks; no collection authority."""
    parents, models, schedules, registries = [], {}, {}, {}
    for pi, parent in enumerate(original_parents):
        scan.check()
        if parent['parent_index'] != pi:
            raise ContractError('ordered original parent index mismatch')
        if game_actions is None:
            model = recorded_model_config_v2(parent['model'], parent['semantics']['token_signature'])
        else:
            from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
            from gf_models.structured_config import model_config_v3
            model = model_config_v3(parent['model'], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
        models[pi] = model
        generations = parent['generations']
        binding = InferenceBindingV2(**parent['execution']['inference']['binding'])
        visibility = parent['execution']['inference']['cuda_visible_devices']
        def policy(number: int, generations: Any = generations, model: Any = model,
                   binding: InferenceBindingV2 = binding, visibility: Any = visibility,
                   parent: Json = parent) -> tuple[str, RuntimePolicyBinding]:
            record = generations[number]
            if game_actions is None:
                label = recorded_policy_label(model, binding, record['inference_sha256'], cuda_visible_devices=visibility)
                options = {'version': '2', 'checkpoint': str(Path(parent['directory']) / 'artifacts' / f'{number:06}'),
                    'expected_manifest_digest': record['inference_sha256'], 'model': asdict(model),
                    'inference': asdict(binding), 'cuda_visible_devices': visibility,
                    'action_contract': asdict(contract), 'purpose': 'component', 'synthetic_capacity': False}
            else:
                from gf_common.card_instance_rules import card_instance_rules
                from gf_models.structured_binding import input_binding_game_v3, policy_label_game_v3
                label = policy_label_game_v3(model, binding, record['inference_sha256'], action_contract=contract,
                    rules=card_instance_rules(), action_config_hash=digest(asdict(game_actions)), cuda_visible_devices=visibility)
                options = {'version': '3', 'checkpoint': str(Path(parent['directory']) / 'artifacts' / f'{number:06}'),
                    'expected_manifest_digest': record['inference_sha256'], 'model': asdict(model),
                    'inference': asdict(binding), 'cuda_visible_devices': visibility,
                    'action_config_hash': digest(asdict(game_actions)), **input_binding_game_v3(contract, card_instance_rules())}
            return label, RuntimePolicyBinding(model.model_id, options)
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
                if game_actions is None:
                    actual = inspect_retained_artifact(directory, expected_inference_digest=generation['inference_sha256'],
                        action_contract=contract, purpose='component', synthetic_capacity=False, scan=scan)
                else:
                    from gf_common.card_instance_rules import card_instance_rules
                    from gf_models.structured_sealed import inspect_retained_game_artifact_v3
                    actual = inspect_retained_game_artifact_v3(directory, expected_inference_digest=generation['inference_sha256'],
                        actions=game_actions, rules=card_instance_rules(), scan=scan)
                if actual.resume is None or scan.files[directory / 'resume.json'][1] != generation['resume_sha256']:
                    raise ContractError('sealed original resume link mismatch')
                if canonical(actual.resume['binding']) != canonical(parent['execution']['learner']['binding']):
                    raise ContractError('sealed original learner execution differs')
            else:
                descriptor = generation['pruned_descriptor']
                if game_actions is None:
                    actual = inspect_pruned_inference(directory,
                        expected_inference_digest=generation['inference_sha256'], expected_resume_digest=generation['resume_sha256'],
                        original_files=descriptor['original_files'], deletion_receipt={
                            'status': 'deleted', 'inference_sha256': generation['inference_sha256'],
                            'original_files': descriptor['original_files'],
                            'deletion_intent': descriptor['deletion_intent'], 'deletion_receipt': descriptor['deletion_receipt']},
                        action_contract=contract, purpose='component', synthetic_capacity=False, scan=scan)
                else:
                    from gf_common.card_instance_rules import card_instance_rules
                    from gf_models.structured_sealed import inspect_pruned_game_inference_v3
                    actual = inspect_pruned_game_inference_v3(directory,
                        expected_inference_digest=generation['inference_sha256'], expected_resume_digest=generation['resume_sha256'],
                        original_files=descriptor['original_files'], deletion_receipt={
                            'status': 'deleted', 'inference_sha256': generation['inference_sha256'],
                            'original_files': descriptor['original_files'], 'deletion_intent': descriptor['deletion_intent'],
                            'deletion_receipt': descriptor['deletion_receipt']}, actions=game_actions, rules=card_instance_rules(), scan=scan)
            if canonical(actual.inference['model']) != canonical(asdict(model)) or digest(actual.files) != generation['artifact_files_sha256']:
                raise ContractError('sealed original full model/resources/file map differs')
            sealed.append({'generation': gi, 'inference_sha256': generation['inference_sha256'],
                'resume_sha256': generation['resume_sha256'], 'artifact_files_sha256': digest(actual.files),
                'retained': actual.retained})
            # Reconstruct all original train schedules with the unchanged helper.
            label, learner = policy(gi)
            registry = {('random-structured-v1' if game_actions is None else 'random-card-instance-v2'): RuntimePolicyBinding('random'),
                        ('heuristic-structured-v1' if game_actions is None else 'heuristic-card-instance-v2'): RuntimePolicyBinding('heuristic'), label: learner}
            snapshots = []
            controls = generation['parent_controls']
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = controls['schedule_opponent_rng']
            for number in controls['snapshot_pool']:
                name, value = policy(number)
                registry[name] = value
                snapshots.append(name)
            groups = training_groups(parent['state']['seed'], gi, label, tuple(snapshots), rng, actions_version=2 if game_actions is None else 3)
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
    return parents, models, schedules, registries


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
    parents, models, schedules, registries = _verify_parent_controls(summary['parents'], contract, scan)
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

        from gf_common.preprocessing import preparation_config
        self.settings = preparation_config(asdict(settings))
        if self.settings.backend != 'rust' or self.settings.reader != 'structured-v1':
            raise ContractError('explicit structured native Source consumer required')
        self._initialize_owner(plan, max_owned_bytes=max_owned_bytes, deadline=deadline,
            extra_owned_bytes=extra_owned_bytes, max_scratch_bytes=max_scratch_bytes, native=native)

    def _initialize_owner(self, plan: Json, *, max_owned_bytes: int, deadline: float,
                          extra_owned_bytes: int, max_scratch_bytes: int | None, native: Any) -> None:
        from contextlib import ExitStack

        from gf_training.structured_rollout import _owned_bytes
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

    def _session_class_name(self) -> str:
        return 'StructuredNativeSession'

    def _before_session_construct(self) -> None:
        """Concrete corrected owner may pin its own binary before native allocation."""

    def _construct_session(self, session_class: Any, plan: bytes, settings: bytes, cancel: Any) -> Any:
        return session_class(plan, settings, self.deadline, cancel)

    def _after_executor_retired(self) -> None:
        """Concrete byte owners retire before the retained parent locks release."""

    def _on_native_failure(self, error: BaseException) -> None:
        """Packed owners fence cleanup; physical ownership preserves its behavior."""

    def _make_executor(self, cancel: Any) -> Any:
        from gf_training.native_execution import NativeExecution
        return NativeExecution(cancel, thread_name_prefix='gf-structured-source')

    def _native_call(self, function: Any, builder: Any = None, *, additional: int = 0,
                     envelope: int = 0, retained: bool = True) -> Any:
        arguments = (self._held(additional),) if builder is None else builder(self._held(additional))
        return self._executor.call(function, *arguments)

    def _construct_owned(self, session_class: Any, cancel: Any) -> None:
        def arguments(held: int) -> tuple[bytes, bytes]:
            return canonical(self.plan).encode(), canonical({'memory_bytes': self.budget, 'scratch_bytes': self.scratch_limit,
                'max_source_file_bytes': self.settings.max_source_file_mib * 1024**2,
                'max_record_bytes': self.settings.max_record_mib * 1024**2,
                'chunk_target_bytes': min(self.settings.chunk_target_mib * 1024**2, 64 * 1024**2)}).encode()
        def construct(plan: bytes, settings: bytes) -> None:
            self._session = self._construct_session(session_class, plan, settings, cancel)
        self._native_call(construct, arguments, retained=False)

    def _control_summary(self) -> bytes:
        return self._executor.call(self._session.control_summary)

    def _authorize_summary(self) -> tuple[Json, SealedScan, dict[int, Any]]:
        assert self.summary is not None
        return _authorize_controls(self.summary, self._session, max_owned_bytes=self.budget, deadline=self.deadline)

    def _authorize_owned(self) -> None:
        if self.plan['mode'] == 'historical_parent':
            self._check_historical()
            self._correspondence_done = True
        from gf_common.artifacts import json_bound
        self._native_call(self._session.authorize, lambda held: (canonical(self._auth).encode(), held),
            envelope=8 * json_bound(self._auth) + 65536)

    def __enter__(self) -> Self:
        from gf_training.lineage import _OWNED_LOCKS, run_lock
        from gf_training.native_backend import binary_identity, load_native
        from gf_training.structured_rollout import _owned_bytes
        try:
            self._check()
            self.native = self.native or load_native()
            session_class = getattr(self.native, self._session_class_name(), None)
            if not callable(session_class):
                raise ContractError('native binary lacks the concrete structured Source session')
            self.identity = binary_identity(self.native)
            self.base += _owned_bytes(self.identity)
            self.budget = self.limit - self.extra - self.base
            if self.budget <= 0:
                raise ContractError('native Source binary/control ownership exhausted before I/O')
            self._before_session_construct()
            for entry in sorted(self.plan['parents'], key=lambda value: value['directory']):
                directory = Path(entry['directory']).resolve()
                if directory not in _OWNED_LOCKS:
                    self._locks.enter_context(run_lock(directory))
            cancel = self.native.CancelHandle(self.deadline)
            self._executor = self._make_executor(cancel)
            self._construct_owned(session_class, cancel)
            encoded = self._control_summary()
            self.summary = __import__('json').loads(encoded)
            self._auth, self.sealed, self.models = self._authorize_summary()
            self._authorize_owned()
            self._check()
            return self
        except BaseException as error:
            self._on_native_failure(error)
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
        self._native_call(self._session.recheck, additional=additional)

    def original_config_inputs(self, parent_index: int, additional: int = 0) -> list[Json]:
        """One bounded management query from this SAME retained original Run2."""
        from gf_common.domain import integer
        self._check()
        assert self.summary is not None
        integer(parent_index, 0, len(self.summary['parents'])-1)
        getter=getattr(self._session,'original_config_inputs',None)
        if not callable(getter):
            raise ContractError('native source lacks original input-origin query before actor preflight')
        raw=__import__('json').loads(self._executor.call(getter,parent_index,self._held(additional)))
        closed(raw,{'schema_version','kind','control_fingerprint','parent_index','run_sha256','config_inputs'})
        parent=self.summary['parents'][parent_index]
        if (type(raw['schema_version']) is not int or raw['schema_version']!=2
                or raw['kind']!='structured-original-config-inputs'
                or type(raw['parent_index']) is not int or raw['parent_index']!=parent_index
                or raw['control_fingerprint']!=self.summary['control_fingerprint']
                or raw['run_sha256']!=parent['run_sha256'] or type(raw['config_inputs']) is not list):
            raise ContractError('original input-origin query binding differs from retained source')
        return raw['config_inputs']

    def begin_update(self) -> None:
        self._check()
        if self.plan['mode'] != 'live_cohort' or not self._correspondence_done or self._begun:
            raise ContractError('actual live original correspondence must precede one update phase')
        self.recheck()  # Both ORIGINAL native and sealed owners, immediately before Adam.
        self._native_call(self._session.begin_update)
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
        try:
            self._after_executor_retired()
        except BaseException as cleanup:  # noqa: BLE001 -- original failure wins, actual owner/locks remain retained.
            if error is None:
                error = cleanup
            else:
                error.add_note('structured Source original-byte retirement: ' + str(cleanup))
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
        initiating = value if isinstance(value, BaseException) else failure
        if initiating is not None:
            self._on_native_failure(initiating)
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

def _payload_views(payload: Any, *, game: bool = False, recorded: bool = False,
                   packed: bool = False, hpo: bool = False) -> tuple[Json, dict[str, Any]]:
    """Validate closed bank descriptors before creating any named NumPy view."""
    import json
    import math

    from gf_common.domain import integer
    if type(payload) is not tuple or len(payload) != 4 or any(type(banks) is not tuple for banks in payload[1:]):
        raise ContractError('closed typed-bank tuple payload required')
    metadata = json.loads(payload[0])
    keys = {'schema_version', 'kind', 'collection_index', 'metadata', 'arrays', 'source_proof', 'correspondence'}
    if not game:
        keys.add('control_fingerprint')
    closed(metadata, keys)
    expected_kind = ('card-instance-recorded-cohort-payload' if recorded else 'card-instance-cohort-payload') if game else 'structured-cohort-payload'
    if hpo:
        expected_kind = 'card-instance-hpo-recorded-cohort-payload' if recorded else 'card-instance-hpo-cohort-payload'
    if (recorded and not game or type(metadata['schema_version']) is not int
            or hpo and not game or metadata['schema_version'] != (3 if hpo else 2 if game else 1) or metadata['kind'] != expected_kind):
        raise ContractError('unsupported structured cohort payload')
    is_packed = metadata['source_proof'].get('kind') == ('card-instance-hpo-packed-original-source-proof' if hpo else 'card-instance-packed-original-source-proof')
    if packed != is_packed or packed and not game:
        raise ContractError('cohort payload packed Source context differs')
    if packed:
        from gf_common.group_pack_game import checked_packed_proof
        physical = metadata['source_proof']
        if hpo:
            from gf_common.hpo import checked_selection
            checked_selection(physical['selection'])
            physical = {key: value for key, value in physical.items() if key not in ('selection', 'expected_optimizer_step_delta')}
            physical = {**physical, 'schema_version': 2, 'kind': 'card-instance-packed-original-source-proof'}
        checked_packed_proof(physical)
        if metadata['correspondence'].get('recipe') != ('card-instance-hpo-packed-source-correspondence-v1' if hpo else 'card-instance-packed-source-correspondence-v1'):
            raise ContractError('packed cohort correspondence recipe differs')
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
