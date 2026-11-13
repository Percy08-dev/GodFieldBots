"""Closed corrected native inputs behind the existing game Source issuer.

A diagnostic historical owner can compare recorded inputs; it cannot issue an
update. Native flags/DTOs/arrays never supply the live Python issuer authority.
"""
from __future__ import annotations

import json
import time
import traceback as traceback_module
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.preprocessing import preparation_config
from gf_models.structured_binding import input_binding_game_v3
from gf_models.structured_sealed import SealedScan
from gf_training.native_structured_session import (
    NativeStructuredSource,
    _attach_gae,
    _payload_views,
    _verify_parent_controls,
)

COLLECTION_FIELDS = {'collection_index', 'directory', 'generation', 'manifest_sha256', 'config_sha256',
    'linkage', 'schedule_sha256', 'bindings_sha256', 'source_proof', 'semantics', 'learner_label',
    'learner_decisions', 'correspondence', 'numeric'}


def _require_final_validation_native(settings: Any, native: Any = None, *, packed: bool = False) -> Any:
    """Full closed capability/API/binary preflight; no parent or Source allocation."""
    from gf_common.card_instance_collection import NATIVE_FINAL_VALIDATION_MARKER
    from gf_common.card_instance_formal import NATIVE_FORMAL_MARKER
    from gf_training.native_backend import (
        binary_identity,
        card_instance_reader_settings,
        load_native,
    )
    module = load_native() if native is None else native
    if settings.backend != 'rust' or settings.reader != 'participant-structured-v2':
        raise ContractError('explicit corrected native final64 reader required')
    if packed:
        from gf_training.native_backend import require_pack_checksum
        require_pack_checksum(module)
    card_instance_reader_settings(settings, module)
    info = dict(module.build_info())
    if (info.get('card_instance_final_validation') != NATIVE_FINAL_VALIDATION_MARKER
            or info.get('card_instance_formal_controls') != NATIVE_FORMAL_MARKER):
        raise ContractError('Rust final64 capability unavailable; formal-control marker is insufficient; no fallback')
    if packed:
        from gf_common.card_instance_collection import NATIVE_PACKED_FINAL_VALIDATION_MARKER
        from gf_training.card_instance_formal import require_native_formal_controls
        require_native_formal_controls(module, packed=True)
        if info.get('card_instance_packed_final_validation') != NATIVE_PACKED_FINAL_VALIDATION_MARKER:
            raise ContractError('Rust packed final64 capability unavailable; no fallback')
    cls = getattr(module, 'CardInstanceNativeSession', None)
    if any(not callable(getattr(cls, name, None)) for name in ('inspect_final_validation',
            'export_final_validation', 'recheck_final_validation', 'close_final_validation')):
        raise ContractError('native final64 four-method ABI is incomplete; no fallback')
    identity = binary_identity(module)
    if identity['build'] != info or identity['gil_disabled'] is not False:
        raise ContractError('native final64 loaded binary/source/ABI differs')
    return module


FINAL_SUMMARY_FIELDS = {'schema_version', 'kind', 'parent_run_sha256', 'directory', 'manifest_kind',
    'manifest_sha256', 'source_set_sha256', 'linkage', 'config_projection', 'validation_invocation',
    'elapsed_seconds', 'games', 'selected_decisions', 'zero_endpoint_games', 'episodes',
    'qualified', 'parent_qualification'}


class _NativeFinalValidationInputs:
    """One private collection loan from the parent's ACTUAL existing issuer owner.

    Only FormalGameParent creates and retains this object. It owns neither a new
    session/executor nor Source/fit permission; Actor64 owns the resource token.
    """
    def __init__(self, owner: _NativeGameInputs, parent: Any, token: object, *, deadline: float) -> None:
        if type(owner) is not _NativeGameInputs:
            raise ContractError('SAME concrete retained corrected native inputs required')
        self._owner, self._parent, self._token = owner, parent, token
        slot = getattr(parent, '_packed_final_originals', None)
        self._provider = None if slot is None else slot.provider
        if getattr(owner, '_artifact_provider', None) is not None and self._provider is None:
            raise ContractError('packed final native loan requires SAME parent-preowned final provider')
        self.deadline = min(deadline, parent.deadline, owner.deadline)
        self._source: Any = None
        self._summary: Json | None = None
        self._packet: Any = None
        self._records: tuple[Any, ...] | None = None
        self._started = self._closed = False
        self._failure: BaseException | None = None

    def bind_source(self, source: Any) -> None:
        from gf_training.card_instance_final_validation import GameFinalValidationSource
        if (self._source is not None or type(source) is not GameFinalValidationSource
                or source._parent is not self._parent or source._native is not self):
            raise ContractError('SAME actual final64 Source cleanup owner required')
        self._source = source

    @property
    def parent_run_sha256(self) -> str:
        self._cheap()
        assert self._owner.summary is not None
        return str(self._owner.summary['parent']['run_sha256'])

    @property
    def aggregate_window(self) -> int:
        self._cheap()
        return int(self._owner.models[0].event.aggregate_window)

    def _admit_local(self, amount: int) -> None:
        loan=getattr(self._parent,'_resource_loan',None)
        if loan is None:self._parent.admit_external(amount)
        else:loan.admit_local(amount)

    def _other_base(self) -> int:
        loan=getattr(self._parent,'_resource_loan',None)
        return 0 if loan is None else loan.base_bytes

    def _cheap(self) -> None:
        if self._failure is not None:
            raise ContractError('native final64 loan is permanently poisoned') from self._failure
        if self._closed:
            raise ContractError('native final64 loan is closed')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('native final64 retained caller deadline')
        self._parent.require_active()
        from gf_training.card_instance_final_validation import _require_borrow
        _require_borrow(self._parent, self._token)
        self._owner._check()

    def _remember(self, error: BaseException) -> None:
        self._failure = self._failure or error
        self._parent._native_final_failure(self, self._failure)
        vars(self._failure)['_game_final_validation_cleanup_owner'] = self._source or self._parent
        vars(self._failure)['_game_final_validation_parent_owner'] = self._parent

    def inspect(self, encoded: bytes) -> Json:
        self._cheap()
        if self._started:
            raise ContractError('native final64 inspection has one lifetime')
        try:
            self._owner.recheck()
            self._started = True  # Reachable through SAME parent/Source before I/O.
            raw = (self._owner._native_call(self._owner._session.inspect_final_validation,
                lambda held: (encoded, held, self._provider), envelope=8*len(encoded)+65536)
                if self._provider is not None else self._owner._executor.call(
                    self._owner._session.inspect_final_validation, encoded, self._owner._held()))
            # Native retains its concrete summary/control reservation before this
            # JSON decode; no second per-parent allowance is constructed.
            packed = self._provider is not None
            summary = closed(json.loads(raw), FINAL_SUMMARY_FIELDS | ({'storage_binding', 'source_proof'} if packed else set()))
            if (type(summary['schema_version']) is not int or summary['schema_version'] != (2 if packed else 1)
                    or summary['kind'] != ('card-instance-packed-final-validation-summary' if packed else 'card-instance-final-validation-summary')
                    or summary['games'] != 64 or summary['qualified'] is not False
                    or summary['parent_qualification'] != 'not_performed'
                    or type(summary['episodes']) is not list or len(summary['episodes']) != 64):
                raise ContractError('native final64 closed structural summary differs')
            if packed:
                assert self._provider is not None
                from gf_common.group_pack_game import checked_packed_proof, checked_storage_binding
                checked_storage_binding(summary['storage_binding'])
                proof = checked_packed_proof(summary['source_proof'])
                description = json.loads(self._provider.describe(summary['directory']))
                if (proof['storage'] != description['storage'] or proof['custody'] != description['custody']
                        or proof['source_set_sha256'] != summary['source_set_sha256']):
                    raise ContractError('packed final summary differs from SAME final originals')
            for i, row in enumerate(summary['episodes']):
                closed(row, {'episode_index', 'episode_id', 'game_id', 'viewer', 'decisions', 'end_kind', 'outcome', 'episode_digest'})
                if type(row['episode_index']) is not int or row['episode_index'] != i or row['end_kind'] != 'natural':
                    raise ContractError('native final64 ordered natural endpoints differ')
                integer(row['viewer'], 1, 2)
                integer(row['decisions'], 0)
            self._summary = summary
            self.recheck()
            return summary
        except BaseException as error:
            self._remember(error)
            raise

    def recheck(self) -> None:
        self._cheap()
        try:
            self._parent.recheck()
            if self._provider is None:
                self._owner._executor.call(self._owner._session.recheck_final_validation, self._owner._held())
            else:
                self._owner._native_call(self._owner._session.recheck_final_validation)
        except BaseException as error:
            self._remember(error)
            raise

    @staticmethod
    def _decode_bound(packet: Any) -> int:
        """Use the actual native prepaid metadata lease; no raw DTO copy here."""
        if type(packet) is not tuple or len(packet) != 4 or type(packet[0]) is not bytes:
            raise ContractError('closed native final64 packet required')
        # publish_raw_banks pre-admits >=64*metadata bytes plus32MiB before
        # returning. This bounded management preview is within that SAME lease.
        metadata = json.loads(packet[0])
        frames = len(metadata['decisions'])
        events = len(metadata['bank']['event_records'])
        uint8 = sum(value.nbytes for value in packet[1])
        all_bytes = sum(value.nbytes for banks in packet[1:] for value in banks)
        # Covers the decoder's metadata/index containers, all owned JSON/codec
        # copies, reference-token temporaries and per-record sizing scratch.
        return (64 * (len(packet[0]) + uint8 + all_bytes)
            + 65536 * (1 + events + 4 * frames)
            + 1024 * (len(metadata['arrays']) + events + frames + 3))

    def _next(self, index: int, *, base_external: int, max_owned_bytes: int, aggregate_window: int) -> Any:
        from gf_training.native_card_instance_final_validation import (
            _decode_final_validation_packet,
        )
        self._cheap()
        assert self._summary is not None
        try:
            self.recheck()
            self._packet = (self._owner._executor.call(self._owner._session.export_final_validation, index, self._owner._held())
                if self._provider is None else self._owner._native_call(self._owner._session.export_final_validation,
                    lambda held: (index, held)))
            bound = self._decode_bound(self._packet)
            self._admit_local(base_external + bound)
            if self._parent.resident_bytes + self._other_base() + base_external + bound > max_owned_bytes:
                raise ContractError('final64 joint decoder admission exhausted before copies')
            def reserve(growth: int) -> None:
                self._cheap()
                if integer(growth, 0) > bound:
                    raise ContractError('native final64 decoder exceeded pre-admitted growth')
            return _decode_final_validation_packet(self._packet, summary=self._summary, episode_index=index,
                actions=self._owner.actions, rules=card_instance_rules(), aggregate_window=aggregate_window,
                max_owned_bytes=max_owned_bytes, retained_owned_bytes=self._parent.resident_bytes + self._other_base() + base_external,
                max_record_bytes=min(64*1024**2, self._owner.settings.max_record_mib*1024**2),
                reserve=reserve, check=self._cheap, _packed_owner=self if self._provider is not None else None)
        except BaseException as error:
            self._remember(error)
            # Retire completed decoder frames' NumPy views before native close;
            # the initiating exception/traceback and SAME cleanup owner survive.
            try:
                traceback_module.clear_frames(error.__traceback__)
            except BaseException:  # noqa: BLE001, S110 - diagnostic/cleanup allocation cannot replace primary
                pass
            raise
        finally:
            self._packet = None

    def records(self, *, base_external: int, max_owned_bytes: int, aggregate_window: int) -> tuple[Any, ...]:
        from gf_training.structured_rollout import _owned_bytes
        if self._records is not None:
            raise ContractError('native final64 records have one materialization')
        rows: list[Any] = []
        try:
            for i in range(64):
                held = base_external + _owned_bytes(rows)
                self._admit_local(held)
                rows.append(self._next(i, base_external=held, max_owned_bytes=max_owned_bytes,
                    aggregate_window=aggregate_window))
            self._admit_local(base_external + _owned_bytes(rows) + 65536)
            self._records = tuple(rows)
            self.recheck()
            return self._records
        except BaseException as error:
            rows.clear()
            self._records = None
            self._remember(error)
            raise

    @property
    def cleanup_complete(self) -> bool:
        return self._closed and self._packet is None and self._records is None

    @property
    def parent_cleanup_complete(self) -> bool:
        """Physical slot plus associated Source buffers/exclusion, without guards."""
        return self.cleanup_complete and (self._source is None or self._source.cleanup_complete)

    def retire_from_parent(self, initiating: BaseException | None = None) -> None:
        if initiating is not None:
            self._failure = self._failure or initiating
        if self._source is not None and not self._source.cleanup_complete:
            self._source._failure = self._source._failure or self._failure
            self._source.close()
        else:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._packet = self._records = None
        failure = self._failure
        if self._started:
            for _ in range(2):
                try:
                    if self._provider is None:
                        self._owner._executor.call(self._owner._session.close_final_validation)
                    else:
                        self._owner._executor.close_final_validation()
                    self._closed = True
                    break
                except BaseException as error:  # noqa: BLE001 - SAME reachable owner confirms cleanup vs error
                    failure = failure or error
                    vars(failure)['_game_final_validation_cleanup_owner'] = self._source or self._parent
                    vars(failure)['_game_final_validation_parent_owner'] = self._parent
                    if failure is not error:
                        try:
                            failure.add_note('native final64 cleanup: ' + str(error))
                        except BaseException:  # noqa: BLE001, S110 - diagnostics never replace first error/owner
                            pass
        else:
            self._closed = True
        if not self._closed:
            failure = failure or ContractError('native final64 cleanup remains unconfirmed')
        else:
            self._summary = None
        if failure is not None:
            self._remember(failure)
            raise failure


def _authorize_game_controls(summary: Json, native: Any, actions: ActionConfigV3, *,
                             max_owned_bytes: int, deadline: float,
                             extra_owned_bytes: int = 0, artifact_provider: Any = None) -> tuple[Json, SealedScan, dict[int, Any]]:
    binding = input_binding_game_v3(actions.action_contract, card_instance_rules())
    packed = artifact_provider is not None
    packed_fields = {'storage_binding', 'artifact_registry_sha256'} if packed else set()
    closed(summary, {'schema_version', 'kind', 'reader_contract', 'mode', 'parent', 'collections',
        'authorization_max_bytes', 'qualified', 'parent_qualification'} | set(binding) | packed_fields)
    if (type(summary['schema_version']) is not int or summary['schema_version'] != (4 if packed else 3)
            or summary['kind'] != ('card-instance-packed-source-controls' if packed else 'card-instance-source-controls')
            or summary['reader_contract'] != 'participant-structured-v2'
            or summary['mode'] not in ('historical_parent', 'live_cohort')
            or summary['qualified'] is not False or summary['parent_qualification'] != 'not_performed'
            or any(canonical(summary[key]) != canonical(value) for key, value in binding.items())):
        raise ContractError('closed corrected native original controls3 required')
    if packed:
        from gf_common.group_pack_game import checked_storage_binding
        checked_storage_binding(summary['storage_binding'])
        if summary['artifact_registry_sha256'] != artifact_provider.registry_sha256:
            raise ContractError('packed native summary registry differs from SAME provider')
    parent = summary['parent']
    closed(parent, {'schema_version', 'kind', 'validation_scope', 'npz_validated', 'journal6_validated',
        'qualified', 'parent_qualification', 'directory', 'state', 'clocks', 'model', 'execution',
        'semantics', 'actions_sha256', 'identity_sha256', 'generations', 'control_fingerprint',
        'parent_index', 'run_sha256', 'run_limits', 'hardware'} | set(binding) | ({'storage_binding'} if packed else set()))
    if (parent['schema_version'] != (4 if packed else 3)
            or parent['kind'] != ('card-instance-packed-parent-controls' if packed else 'card-instance-parent-controls')
            or parent['validation_scope'] != 'metadata-maps-filehashes-only'
            or any(parent[key] is not False for key in ('npz_validated', 'journal6_validated', 'qualified'))
            or parent['parent_qualification'] != 'not_performed' or parent['parent_index'] != 0
            or parent['actions_sha256'] != digest(asdict(actions))
            or any(canonical(parent[key]) != canonical(value) for key, value in binding.items())):
        raise ContractError('native parent3 context/scope differs')
    if packed and (parent['storage_binding'] != summary['storage_binding']
            or parent['state'].get('schema_version') != 4 or parent['state'].get('storage_binding') != summary['storage_binding']):
        raise ContractError('packed native parent state/binding differs')
    remaining = max_owned_bytes - native.used_bytes() - extra_owned_bytes
    if remaining <= 0:
        raise ContractError('joint native/sealed ownership exhausted before game inspection')
    scan = SealedScan(max_owned_bytes=remaining, deadline=deadline)
    if 'run_variant' in parent['state']:
        from gf_common.card_instance_formal import FORMAL_RUN_FIELDS
        from gf_training.card_instance_formal import read_formal_control
        original = scan.object(Path(parent['directory'])/'run.json', parent['run_sha256'])
        read_formal_control(Path(parent['directory']), original, scan)
        if any(canonical(parent['state'].get(key)) != canonical(original[key]) for key in FORMAL_RUN_FIELDS):
            raise ContractError('native formal control summary differs from SAME original Run3')
    parents, models, schedules, registries = _verify_parent_controls([parent], actions.action_contract,
        scan, game_actions=actions)
    collections = []
    if type(summary['collections']) is not list:
        raise ContractError('ordered original game collection summaries required')
    for index, collection in enumerate(summary['collections']):
        scan.check()
        closed(collection, COLLECTION_FIELDS)
        if packed:
            from gf_common.group_pack_game import checked_packed_proof
            proof = checked_packed_proof(collection['source_proof'])
            description = json.loads(artifact_provider.describe(collection['directory']))
            if (proof['storage'] != description['storage'] or proof['custody'] != description['custody']
                    or collection['correspondence']['recipe'] != 'card-instance-packed-source-correspondence-v1'):
                raise ContractError('packed native collection differs from SAME originals')
        generation = integer(collection['generation'], 0, len(parent['generations']) - 1)
        if (collection['collection_index'] != index or collection['config_sha256'] != parent['state']['config_sha256']
                or collection['schedule_sha256'] != schedules[0, generation]
                or collection['bindings_sha256'] != registries[0, generation]):
            raise ContractError('native original game schedule/registry/Config differs')
        link = closed(collection['linkage'], {'parent_run_id', 'parent_run_path', 'batch_id', 'collection_generation',
            'learner_seats', 'schedule_digest', 'policy_registry_digest'})
        if (link['parent_run_id'] != parent['state']['run_id'] or link['parent_run_path'] != parent['directory']
                or link['collection_generation'] != generation or link['batch_id'] != f'batch-{generation:06}'
                or link['schedule_digest'] != schedules[0, generation]
                or link['policy_registry_digest'] != registries[0, generation]):
            raise ContractError('native original game linkage differs')
        numeric = collection['numeric']
        if numeric is not None:
            closed(numeric, {'directory', 'sha256', 'files'})
        collections.append({'collection_index': index, 'manifest_sha256': collection['manifest_sha256'],
            'expected_schedule_sha256': schedules[0, generation], 'expected_bindings_sha256': registries[0, generation],
            'source_set_sha256': collection['source_proof']['source_set_sha256'],
            'correspondence_sha256': collection['correspondence']['binding_sha256'],
            'numeric_sha256': None if numeric is None else numeric['sha256']})
    scan.finalize()
    return {'schema_version': 4 if packed else 3, 'summary_sha256': digest(summary), 'parent': parents[0],
            'collections': collections, **({key: summary[key] for key in packed_fields})}, scan, models


class _NativeGameInputs(NativeStructuredSource):
    """Concrete inputs representation only; SAME _GameBatchSource issues authority."""
    _explicit_candidate: bool

    def __init__(self, plan: Json, settings: Any, *, actions: ActionConfigV3, max_owned_bytes: int,
                 deadline: float, extra_owned_bytes: int = 0, max_scratch_bytes: int | None = None,
                 native: Any = None, collection_index: int = 0, artifact_provider: Any = None, shared_controls: Any = None) -> None:
        self.actions = actions
        self._hpo = type(plan.get('schema_version')) is int and plan['schema_version'] == 5
        self._hpo_expected: Json | None = None
        input_binding_game_v3(actions.action_contract, card_instance_rules())
        self.settings = preparation_config(asdict(settings))
        if self.settings.backend != 'rust' or self.settings.reader != 'participant-structured-v2':
            raise ContractError('explicit corrected native inputs required')
        self._initialize_owner(plan, max_owned_bytes=max_owned_bytes, deadline=deadline,
            extra_owned_bytes=extra_owned_bytes, max_scratch_bytes=max_scratch_bytes, native=native)
        from gf_training.native_group_pack import OriginalArtifactProvider
        self._artifact_provider = artifact_provider
        self._shared_controls = shared_controls
        self._last_held = 0
        if artifact_provider is not None:
            if (type(artifact_provider) is not OriginalArtifactProvider or plan.get('schema_version') != (5 if self._hpo else 4)
                    or plan.get('artifact_registry_sha256') != artifact_provider.registry_sha256):
                raise ContractError('plan4 requires SAME concrete registered original provider')
            from gf_common.group_pack_game import checked_storage_binding
            checked_storage_binding(plan.get('storage_binding'))
            if artifact_provider._empty_controls is not None and (
                    shared_controls is not artifact_provider._empty_controls
                    or plan.get('mode') != 'historical_parent' or plan.get('collections') != []
                    or len(plan.get('parents', [])) != 1):
                raise ContractError('empty provider requires SAME actual historical generation0 controls')
            if shared_controls is not None:
                from gf_training.structured_lineage import _ChainScan
                if (type(shared_controls) is not _ChainScan or shared_controls._packed_pool is not artifact_provider.pool
                        or shared_controls.root != Path(plan['parents'][0]['directory']).resolve()
                        or shared_controls._pool_control_charge < shared_controls.retained
                        or extra_owned_bytes < shared_controls.retained):
                    raise ContractError('actual SAME pool-backed parent ControlScan required for credit')
                self.extra -= shared_controls.retained
            from gf_common.artifacts import json_bound
            self.base += 8 * json_bound(plan) + 131072
            self.extra += artifact_provider.external_bytes
            self.budget = self.limit - self.extra - self.base
            if self.budget <= 0:
                raise ContractError('native original provider ownership exhausted before FFI')
        elif plan.get('schema_version') == 4:
            raise ContractError('plan4 requires explicit original byte provider')
        elif shared_controls is not None:
            raise ContractError('shared packed ControlScan requires an actual artifact provider')
        self._explicit_candidate = native is not None
        self._binary_scan: SealedScan | None = None
        self._binary_identity: bytes | None = None
        self._live_run_sha256: str | None = None
        self.collection_index = collection_index
        self._prepared = False
        self._caller_growth = 0
        self._external_growth = 0
        self._final_working_delta = 0
        self._final_artifact_provider: Any = None
        self._numeric_files: Json | None = None
        self._prepared_receipt: Json | None = None
        self._hpo_save_phase = 'empty'
        self._hpo_save_error: BaseException | None = None

    def _poison_hpo_save(self, error: BaseException) -> None:
        self._hpo_save_phase = 'failed'
        if self._hpo_save_error is None:
            self._hpo_save_error = error

    def _admit_hpo_saved_output(self, projection: Any, generation: int, files: Json) -> None:
        try:
            self._check()
            from gf_training.card_instance_source import IssuedGameProjection
            from gf_training.lineage import checkpoint_path, require_hpo_run_owner
            from gf_training.structured_lineage import _admit_hpo_artifact_append, _ChainScan
            if (not self._hpo or self.plan['mode'] != 'live_cohort' or not self._begun
                    or self._prepared or self._hpo_save_phase != 'empty'
                    or type(projection) is not IssuedGameProjection
                    or projection.source.native_inputs is not self
                    or projection._hpo_save_phase != 'admitting'
                    or type(self.sealed) is not _ChainScan
                    or generation != self.collection['generation'] + 1):
                raise ContractError('SAME live native capsule and acknowledged next save required')
            require_hpo_run_owner(self.sealed.root, projection.source.online.ownership)
            ack = projection._require_hpo_save_ack(projection.process, generation, files)
            self._hpo_save_phase = 'admitting'
            if self._binary_scan is not None:
                self._binary_scan.finalize()
            # The existing native owner independently enforces its armed exact
            # one-next-directory permit before Python changes its frozen view.
            self._native_call(self._session.recheck)
            _admit_hpo_artifact_append(self.sealed, generation)
            from gf_models.structured_artifacts import adam_recipe
            from gf_models.structured_sealed import inspect_retained_game_artifact_v4
            from gf_training.hpo_config import hpo_training_config
            assert self.summary is not None
            state = self.summary['parent']['state']
            recipe = hpo_training_config(state['effective_config']['hpo_spec']['training_recipe'])
            artifact = inspect_retained_game_artifact_v4(checkpoint_path(self.sealed.root, generation),
                expected_inference_digest=ack['artifact']['inference_digest'], actions=self.actions,
                rules=card_instance_rules(), training_recipe=asdict(recipe),
                optimizer_recipe=adam_recipe(recipe.adam_learning_rate, recipe.adam_beta1,
                    recipe.adam_beta2, recipe.adam_epsilon), scan=self.sealed)
            if (canonical(artifact.files) != canonical(files) or artifact.resume is None
                    or artifact.resume['optimizer_steps'] != ack['artifact']['optimizer_steps']
                    or artifact.resume['resume_identity'] != state['resume_identity']
                    or canonical(artifact.resume['model']) != canonical(asdict(self.models[0]))
                    or canonical(artifact.resume['binding']) != canonical(state['execution']['learner']['binding'])):
                raise ContractError('native saved pair differs from the actual child ACK/model/steps')
            self.sealed.finalize()
            if self._binary_scan is not None:
                self._binary_scan.finalize()
            self._hpo_save_phase = 'admitted'
        except BaseException as error:
            self._poison_hpo_save(error)
            raise

    def __enter__(self) -> Self:
        from gf_training.native_backend import (
            binary_identity,
            card_instance_reader_settings,
            load_native,
            recorded_card_instance_preprocessing,
        )
        from gf_training.structured_rollout import _owned_bytes
        self._check()
        mode = self.plan.get('mode')
        if mode not in ('historical_parent', 'live_cohort'):
            raise ContractError('corrected native plan mode differs')
        if mode == 'live_cohort' and self._explicit_candidate:
            raise ContractError('explicit native candidate is historical diagnostic only')
        module = self.native if self._explicit_candidate else load_native()
        if self._artifact_provider is not None:
            from gf_training.native_backend import require_pack_checksum
            require_pack_checksum(module)
        if not self._explicit_candidate:
            if self._hpo:
                from gf_training.native_backend import hpo_reader_settings
                hpo_reader_settings(self.settings, module, packed=self._artifact_provider is not None)
            else:
                card_instance_reader_settings(self.settings, module)
            if self._artifact_provider is not None:
                from gf_training.native_backend import group_pack_reader_settings
                group_pack_reader_settings(self.settings, module)
        current = binary_identity(module)
        evidence = SealedScan(max_owned_bytes=self.budget, deadline=self.deadline)
        binary_path = Path(current['extension_path'])
        with evidence.file(binary_path, current['extension_sha256']):
            pass
        if mode == 'live_cohort':
            parents = self.plan.get('parents')
            if type(parents) is not list or len(parents) != 1:
                raise ContractError('live native provider requires one SAME original parent')
            parent = Path(closed(parents[0], {'directory', 'require_complete'})['directory']).resolve()
            state = evidence.object(parent / 'run.json')
            preprocessing = state.get('identity', {}).get('preprocessing')
            if type(preprocessing) is not dict:
                raise ContractError('live parent lacks recorded native preprocessing identity')
            recorded_card_instance_preprocessing(preprocessing,
                binding=input_binding_game_v3(self.actions.action_contract, card_instance_rules()))
            if canonical(preprocessing['binary']) != canonical(current):
                raise ContractError('live native binary/source identity differs from SAME original parent')
            self._live_run_sha256 = evidence.files[parent / 'run.json'][1]
            del state, preprocessing
        evidence.finalize()
        self._binary_scan = evidence
        self._binary_identity = canonical(current).encode()
        # Exclude persistent preflight evidence from the native budget BEFORE
        # construction; subsequent held/decode reservations must not count it twice.
        self.extra += evidence.peak + _owned_bytes((self._binary_identity, self._live_run_sha256))
        self.native = module
        return super().__enter__()

    def _before_session_construct(self) -> None:
        if self._binary_identity is None or canonical(self.identity).encode() != self._binary_identity:
            raise ContractError('native binary/source changed before Source allocation')
        assert self._binary_scan is not None
        self._binary_scan.finalize()

    def _make_executor(self, cancel: Any) -> Any:
        if self._artifact_provider is None:
            return super()._make_executor(cancel)
        from gf_training.native_group_pack import _PackedNativeExecution
        executor = _PackedNativeExecution(cancel, self._artifact_provider)
        self._executor = executor  # Adopt before the final prepaid N calculation.
        growth = self._artifact_provider.pool.budget.used - self._artifact_provider.baseline_bytes
        if growth < 0:
            raise ContractError('constructor original account shrank without owned retirement')
        self.extra += growth
        self._artifact_provider.baseline_bytes += growth
        self._artifact_provider.external_bytes += growth
        self.budget = self.limit - self.extra - self.base
        if self.budget <= 0:
            raise ContractError('packed constructor account exhausted BEFORE encoding')
        return executor

    def _on_native_failure(self, error: BaseException) -> None:
        pending_scan = getattr(error, '_hpo_authorization_cleanup_scan', None)
        if pending_scan is not None and self.sealed is None:
            self.sealed = pending_scan
        if self._artifact_provider is not None:
            self._artifact_provider._remember(error)
            if self._executor is not None:
                self._executor._remember(error)

    def _check(self) -> None:
        failure = getattr(self, '_hpo_save_error', None)
        if failure is not None:
            raise failure
        super()._check()
        if getattr(self, '_artifact_provider', None) is not None:
            self._artifact_provider._check()

    def _native_call(self, function: Any, builder: Any = None, *, additional: int = 0,
                     envelope: int = 0, retained: bool = True) -> Any:
        if self._artifact_provider is None:
            return super()._native_call(function, builder, additional=additional, envelope=envelope, retained=retained)
        try:
            self._executor.prepare()
            dynamic = max(0, self._artifact_provider.pool.budget.used - self._artifact_provider.baseline_bytes)
            held = dynamic + self._held(additional) + envelope
            used = 0 if self._session is None else int(self._session.used_bytes()) - self._last_held
            if used < 0 or used + held > self.budget:
                raise ContractError('joint packed native ownership exhausted BEFORE argument construction')
            arguments = (held,) if builder is None else builder(held)
            result = self._executor.call(function, *arguments)
            if retained:
                self._last_held = held
            return result
        except BaseException as error:
            self._executor._remember(error)
            raise

    def _attach_final_artifacts(self, provider: Any) -> None:
        from gf_training.native_group_pack import OriginalArtifactProvider
        if (type(provider) is not OriginalArtifactProvider or self._artifact_provider is None
                or provider.pool is not self._artifact_provider.pool or self._final_artifact_provider is not None):
            raise ContractError('one final provider on SAME packed actual parent required')
        self._executor.prepare()
        self._final_artifact_provider = provider
        self._final_working_delta = max(0, provider.working_bytes-self._artifact_provider.working_bytes)
        self._executor.attach_final_provider(provider)
        # Cumulative metadata plus only the positive max-one-group difference
        # is checked through the SAME native caller lease before any final FFI.
        self._native_call(self._session.recheck)

    def _control_summary(self) -> bytes:
        if self._artifact_provider is None:
            return super()._control_summary()
        return self._native_call(self._session.control_summary)

    def close(self) -> None:
        if self._artifact_provider is None:
            return super().close()
        if self._closed:
            return
        primary: BaseException | None = None
        if self._executor is not None:
            try:
                self._executor.close()
            except BaseException as error:
                primary = error
                if not self._executor._closed:
                    vars(error)['_structured_source_cleanup_owner'] = self
                    raise
            self._session_closed = self._executor._native_retired
        try:
            self._artifact_provider.close()
        except BaseException as error:
            vars(error)['_structured_source_cleanup_owner'] = self
            if primary is not None:
                primary.add_note('provider retirement: ' + str(error))
                raise primary
            raise
        self._locks.close()
        self._closed = True
        if primary is not None:
            raise primary

    def _construct_session(self, session_class: Any, plan: bytes, settings: bytes, cancel: Any) -> Any:
        if self._artifact_provider is None:
            return super()._construct_session(session_class, plan, settings, cancel)
        constructor = getattr(session_class, 'from_artifacts', None)
        if not callable(constructor):
            raise ContractError('native packed provider constructor unavailable; no fallback')
        session = constructor(plan, self._artifact_provider, settings, self.deadline, cancel)
        self._executor.bind_native(session)
        return session

    def _after_executor_retired(self) -> None:
        if self._artifact_provider is not None:
            self._artifact_provider.close()

    def recheck(self, additional: int = 0) -> None:
        self._check()
        if self._binary_scan is not None:
            self._binary_scan.finalize()
        super().recheck(additional)
        if self._binary_scan is not None:
            self._binary_scan.finalize()

    def original_config_inputs(self, parent_index: int, additional: int = 0) -> list[Json]:
        self._check()
        integer(parent_index, 0, 0)
        assert self.summary is not None
        getter = getattr(self._session, 'original_config_inputs', None)
        if not callable(getter):
            raise ContractError('native corrected source lacks original input-origin query')
        raw = json.loads(self._native_call(getter, lambda held: (parent_index, held), additional=additional))
        closed(raw, {'schema_version', 'kind', 'summary_sha256', 'parent_index', 'run_sha256', 'config_inputs'})
        if (type(raw['schema_version']) is not int or raw['schema_version'] != 3
                or raw['kind'] != 'card-instance-original-config-inputs'
                or type(raw['parent_index']) is not int or raw['parent_index'] != 0
                or raw['summary_sha256'] != digest(self.summary)
                or raw['run_sha256'] != self.summary['parent']['run_sha256']
                or type(raw['config_inputs']) is not list):
            raise ContractError('corrected original input-origin query differs from retained Source')
        return raw['config_inputs']

    @property
    def cleanup_complete(self) -> bool:
        """Concrete cleanup confirmation, available even when failed/expired."""
        return (self._closed and (self._session is None or self._session_closed)
            and (self._executor is None or (self._executor._closed and self._executor._active is None))
            and (self._artifact_provider is None or self._artifact_provider._closed))

    @property
    def joint_owned_bytes(self) -> int:
        self._check()
        return self.extra+self.base+(0 if self._session is None else int(self._session.used_bytes()))

    def _session_class_name(self) -> str:
        return 'CardInstanceNativeSession'

    def _authorize_summary(self) -> tuple[Json, SealedScan, dict[int, Any]]:
        assert self.summary is not None
        if not self._hpo and self._live_run_sha256 is not None and self.summary['parent']['run_sha256'] != self._live_run_sha256:
            raise ContractError('native live Source differs from SAME preflight parent state')
        if self._hpo:
            from gf_training.native_hpo import authorize_controls
            authorization, scan, models, expected = authorize_controls(self.summary, self._session, self.actions,
                directory=Path(self.plan['parents'][0]['directory']).resolve(), max_owned_bytes=self.budget,
                deadline=self.deadline, artifact_provider=self._artifact_provider)
            self._hpo_expected = expected
            if self._live_run_sha256 is not None and scan.verified_digest(Path(self.plan['parents'][0]['directory']) / 'run.json') != self._live_run_sha256:
                raise ContractError('HPO live facts differ from SAME binary-preflight parent')
            return authorization, scan, models
        return _authorize_game_controls(self.summary, self._session, self.actions,
            max_owned_bytes=self.budget, deadline=self.deadline, artifact_provider=self._artifact_provider)

    def _authorize_owned(self) -> None:
        if not self._hpo:
            return super()._authorize_owned()
        from gf_common.artifacts import json_bound
        raw = self._native_call(self._session.authorize_hpo,
            lambda held: (canonical(self._auth).encode(), held), envelope=8 * json_bound(self._auth) + 65536)
        summary = json.loads(raw)
        if canonical(summary) != canonical(self._hpo_expected):
            raise ContractError('native final HPO proof/selection/correspondence differs from independently replayed originals')
        self.summary = summary
        self._hpo_expected = None
        self._auth = {}
        if self.plan['mode'] == 'historical_parent':
            self._check_historical()
            self._correspondence_done = True

    @property
    def collection(self) -> Json:
        assert self.summary is not None
        integer(self.collection_index, 0, len(self.summary['collections']) - 1)
        return self.summary['collections'][self.collection_index]

    @property
    def proof(self) -> Json:
        return self.collection['source_proof']

    @property
    def correspondence(self) -> Json:
        return self.collection['correspondence']

    @property
    def decision_count(self) -> int:
        return int(self.collection['learner_decisions'])

    def _held(self, additional: int = 0) -> int:
        growth = self._caller_growth
        if self._shared_controls is not None:
            if (self._shared_controls._packed_pool is not self._artifact_provider.pool
                    or self._shared_controls._pool_control_charge < self._shared_controls.retained):
                raise ContractError('SAME ControlScan pool credit no longer covers retained controls')
            growth = 0
        return (super()._held(additional) + growth + self._external_growth + self._final_working_delta)

    def caller_growth(self, value: int, *, external_growth: int = 0) -> None:
        self._caller_growth = integer(value, 0)
        self._external_growth = integer(external_growth, 0)
        self._check()

    def _original_episode(self, collection_index: int, episode_index: int, *, parent: Any,
            budget_loan: Any, episode_id: str, viewer: int, aggregate_window: int,
            base_local: int) -> tuple[Any, Any]:
        """Actual-parent private raw16 transport; no session/summary constructor."""
        from gf_training.native_card_instance_final_validation import (
            _decode_recorded_episode_packet,
        )
        budget_loan.require(parent)
        self._check_historical()
        packet = None
        try:
            self.recheck()
            packet = self._native_call(self._session.export_recorded_episode_payload,
                lambda held: (collection_index, episode_index, held))
            bound = _NativeFinalValidationInputs._decode_bound(packet)
            budget_loan.admit_local(base_local + bound)
            def cheap() -> None:
                self._check()
                parent.require_active()
                budget_loan.require(parent)
            def reserve(n: int) -> None:
                cheap()
                if integer(n, 0) > bound:
                    raise ContractError('original-history decoder exceeds prior local reservation')
            assert self.summary is not None
            record = _decode_recorded_episode_packet(packet, summary=self.summary,
                collection_index=collection_index, episode_index=episode_index,
                episode_id=episode_id, viewer=viewer, actions=self.actions, rules=card_instance_rules(),
                aggregate_window=aggregate_window, max_owned_bytes=parent.limit,
                retained_owned_bytes=parent.resident_bytes+budget_loan.base_bytes+base_local,
                max_record_bytes=min(64*1024**2,self.settings.max_record_mib*1024**2),
                reserve=reserve,check=cheap)
            return packet, record
        except BaseException as error:
            parent._poison(error)
            try:
                traceback_module.clear_frames(error.__traceback__)
            except BaseException:  # noqa: BLE001, S110 - first error/actual cleanup owner survive
                pass
            packet = None
            raise

    def _read_game(self, directory: Path, expected_digest: str, collection: Json) -> Any:
        from gf_training.structured_rollout import read_game_rollout
        self.recheck()
        result: Any
        if self._hpo:
            from gf_training.hpo_config import hpo_training_config
            from gf_training.hpo_rollout import read_hpo_rollout
            assert self.summary is not None
            training = hpo_training_config(self.summary['parent']['state']['effective_config']['hpo_spec']['training_recipe'])
            native_owned = self._session.used_bytes()
            numeric_limit = self.summary['parent']['state']['effective_config']['hpo_spec']['operating']['numeric_budget_bytes']
            result = read_hpo_rollout(directory, expected_digest, self.models[0], training, actions=self.actions,
                rules=card_instance_rules(), expected_selection=collection['selection'],
                expected_correspondence=collection['correspondence'], max_owned_bytes=min(self.budget, native_owned + numeric_limit),
                extra_owned_bytes=native_owned, numeric_budget_bytes=numeric_limit,
                check=self._check, reserve=self._reserve_numeric,
                storage_binding=self.plan.get('storage_binding'))
        else:
            result = read_game_rollout(directory, expected_digest, self.models[0], actions=self.actions,
            rules=card_instance_rules(), declaration=collection['semantics'],
            expected_correspondence=collection['correspondence'], max_owned_bytes=self.budget,
            extra_owned_bytes=self._session.used_bytes(), check=self._check, reserve=self._reserve_decode,
            storage_binding=self.plan.get('storage_binding'))
        expected_count = collection['selection']['used_decisions'] if self._hpo else collection['learner_decisions']
        if len(result) != expected_count or result.policy_version != collection['learner_label']:
            raise ContractError('numeric4 differs from original game endpoints/policy')
        return result

    def _reserve_numeric(self, size: int) -> None:
        self._reserve_decode(size)
        if self._hpo:
            assert self.summary is not None
            limit = self.summary['parent']['state']['effective_config']['hpo_spec']['operating']['numeric_budget_bytes']
            native_working = self._session.numeric_used_bytes()
            if size + (native_working or 0) > limit:
                raise ContractError('HPO incremental numeric working budget exhausted before Python allocation')

    def _check_historical(self) -> None:
        assert self.summary is not None
        for collection in self.summary['collections']:
            numeric = collection['numeric']
            if numeric is None:
                raise ContractError('historical native game collection lacks committed numeric4')
            result = self._read_game(Path(numeric['directory']), numeric['sha256'], collection)
            result.require_learning()
            del result
        self.recheck()

    def read_numeric(self, directory: Path, expected_digest: str) -> Any:
        return self._read_game(directory, expected_digest, self.collection)

    def files_for(self, directory: Path, expected_digest: str) -> Json:
        if self._numeric_files is not None and self._numeric_files['rollout.json'] == expected_digest:
            return dict(self._numeric_files)
        numeric = self.collection['numeric']
        if numeric is None or Path(numeric['directory']) != directory or numeric['sha256'] != expected_digest:
            raise ContractError('numeric files differ from retained native projection')
        return dict(numeric['files'])

    def _publish_packet(self, directory: Path, *, recorded: bool, index: int) -> Json:
        from gf_models.arrays import sha256
        from gf_training.structured_rollout import _owned_bytes, _write_prepared_game_payload
        self._check()
        if self.plan['mode'] != ('historical_parent' if recorded else 'live_cohort'):
            raise ContractError('native packet use/mode differs')
        if recorded:
            destination = directory.resolve()
            assert self.summary is not None
            originals = [Path(self.summary['parent']['directory']).resolve(), *(
                Path(row['directory']).resolve() for row in self.summary['collections'])]
            if any(destination == root or destination.is_relative_to(root) for root in originals):
                raise ContractError('recorded comparison output must be outside every original')
            if directory.exists() or directory.is_symlink():
                raise ContractError('recorded comparison requires a fresh assessment path')
            payload = self._native_call(self._session.export_recorded_cohort, lambda held: (index, held))
        else:
            assert self.summary is not None
            collection = self.summary['collections'][index]
            generation = integer(collection['generation'], 0, 2**63 - 1 if self._hpo else 999)
            expected = Path(self.summary['parent']['directory']) / 'numeric' / f'batch-{generation:06}'
            if directory.absolute() != expected:
                raise ContractError('live numeric publication differs from derived SAME parent/batch target')
            if directory.exists() or directory.is_symlink():
                raise ContractError('live numeric publication requires a fresh nonsymlink target')
            if self._published is not None:
                raise ContractError('one native live numeric publication required')
            payload = self._native_call(self._session.export_cohort)
        metadata, arrays = _payload_views(payload, game=True, recorded=recorded, packed=self._artifact_provider is not None, hpo=self._hpo)
        assert self.summary is not None
        collection = self.summary['collections'][index]
        if (metadata['collection_index'] != index
                or canonical(metadata['source_proof']) != canonical(collection['source_proof'])
                or canonical(metadata['correspondence']) != canonical(collection['correspondence'])):
            raise ContractError('native packet differs from retained original game Source')
        self._reserve_numeric(65536 + 128 * (sum(len(episode['frames']) for episode in metadata['metadata']['episodes'])))
        if self._hpo:
            from gf_training.native_hpo import attach_gae
            labels = attach_gae(metadata['metadata'], arrays)
            from gf_training.structured_rollout import _RolloutContext, _write_prepared_payload
            context = {key: metadata['metadata'][key] for key in ('training_recipe', 'selection', 'expected_optimizer_step_delta')}
            manifest = _write_prepared_payload(directory, metadata['metadata'], arrays, self.models[0],
                resident_bytes=_owned_bytes((self.models[0], labels)), extra_owned_bytes=self._session.used_bytes(),
                max_owned_bytes=self.budget, shard_bytes=min(self.settings.chunk_target_mib * 1024**2, 64 * 1024**2),
                check=self._check, reserve=self._reserve_numeric, _context=_RolloutContext(self.actions, card_instance_rules()), _hpo=context)
        else:
            labels = _attach_gae(metadata['metadata'], arrays)
            manifest = _write_prepared_game_payload(directory, metadata['metadata'], arrays, self.models[0],
            actions=self.actions, rules=card_instance_rules(), resident_bytes=_owned_bytes((self.models[0], labels)),
            extra_owned_bytes=self._session.used_bytes(), max_owned_bytes=self.budget,
            shard_bytes=min(self.settings.chunk_target_mib * 1024**2, 64 * 1024**2),
            check=self._check, reserve=self._reserve_decode)
        digest_value = sha256(directory / 'rollout.json', self.deadline)
        files = {'rollout.json': digest_value, **manifest['data']['files']}
        del labels, arrays, metadata, payload
        self._native_call(self._session.retire_payload)
        if not recorded:
            self._native_call(self._session.bind_numeric, lambda held: (canonical({'directory': str(directory.absolute()),
                'manifest_sha256': digest_value, 'files': files}).encode(), held),
                envelope=8 * __import__('gf_common.artifacts', fromlist=['json_bound']).json_bound(files) + 65536)
            self._published, self._numeric_files = manifest, files
            self._persistent_python = _owned_bytes((manifest, files))
        result = self._read_game(directory, digest_value, collection)
        result.require_learning()
        del result
        self.recheck()
        self._correspondence_done = True
        return manifest

    def publish_numeric(self, directory: Path) -> Json:
        return self._publish_packet(directory, recorded=False, index=self.collection_index)

    def compare_recorded(self, index: int, directory: Path) -> Json:
        assert self.summary is not None
        integer(index, 0, len(self.summary['collections']) - 1)
        return self._publish_packet(directory, recorded=True, index=index)

    def finalize(self) -> Json:
        self._check()
        if self.plan['mode'] != 'historical_parent' or not self._correspondence_done:
            raise ContractError('only fully checked historical input can finalize read-only')
        self.recheck()
        result = json.loads(self._native_call(self._session.finalize_read))
        closed(result, {'schema_version', 'kind', 'summary_sha256', 'qualified', 'parent_qualification'})
        packed = self._artifact_provider is not None
        if (type(result['schema_version']) is not int or result['schema_version'] != (3 if self._hpo else 2 if packed else 1)
                or result['kind'] != (('card-instance-hpo-packed-source-checked' if packed else 'card-instance-hpo-source-checked') if self._hpo else ('card-instance-packed-source-read-finalized' if packed else 'card-instance-source-read-finalized'))
                or result['summary_sha256'] != digest(self.summary) or result['qualified'] is not False
                or result['parent_qualification'] != 'not_performed'):
            raise ContractError('native finalized receipt differs from SAME retained Source')
        if self.sealed is not None:
            self.sealed.finalize()
        if self._binary_scan is not None:
            self._binary_scan.finalize()
        return result

    def prepare_commit(self) -> None:
        self._check()
        if self._hpo and self._hpo_save_phase != 'admitted':
            raise ContractError('native HPO preparation requires acknowledged saved-output admission')
        if self.plan['mode'] != 'live_cohort' or not self._begun or self._numeric_files is None:
            raise ContractError('native preparation requires its completed live projection')
        self.recheck()
        receipt = json.loads(self._native_call(self._session.prepare_commit))
        closed(receipt, {'schema_version', 'kind', 'summary_sha256', 'collection_manifest_sha256',
            'source_correspondence', 'numeric_manifest_sha256', 'numeric_files_sha256', 'optimizer_steps'}
            | ({'collected_decisions', 'used_decisions', 'expected_optimizer_step_delta', 'selection_sha256'} if self._hpo else set()))
        assert self.summary is not None
        delta = self.proof['expected_optimizer_step_delta'] if self._hpo else 4 * ((self.decision_count + 127) // 128)
        expected_steps = self.summary['parent']['state']['optimizer_steps'] + delta
        if self._hpo and (receipt['collected_decisions'] != self.decision_count
                or receipt['used_decisions'] != self.collection['selection']['used_decisions']
                or receipt['expected_optimizer_step_delta'] != delta
                or receipt['selection_sha256'] != digest(self.collection['selection'])):
            raise ContractError('native HPO prepared receipt differs from frozen selected/collected/delta')
        packed = self._artifact_provider is not None
        if (receipt['schema_version'] != (3 if self._hpo else 2 if packed else 1) or receipt['summary_sha256'] != digest(self.summary)
                or receipt['optimizer_steps'] != expected_steps
                or receipt['collection_manifest_sha256'] != self.proof['manifest_sha256']
                or receipt.get('kind') != (('card-instance-hpo-packed-source-prepared' if packed else 'card-instance-hpo-source-prepared') if self._hpo else ('card-instance-packed-source-prepared' if packed else 'card-instance-source-prepared'))
                or receipt.get('numeric_manifest_sha256') != self._numeric_files['rollout.json']
                or receipt.get('numeric_files_sha256') != digest(dict(sorted(self._numeric_files.items())))
                or canonical(receipt.get('source_correspondence')) != canonical(self.correspondence)):
            raise ContractError('native prepared receipt differs from retained projection')
        if self.sealed is not None:
            self.sealed.finalize()
        if self._binary_scan is not None:
            self._binary_scan.finalize()
        self._session_closed = True
        self._executor.close()
        if self._artifact_provider is not None:
            self._artifact_provider.close()
        self._locks.close()
        self._prepared_receipt = receipt
        self._prepared = True

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        if self._prepared:
            # All reportable work was completed before the same issuer prepared.
            self._session = self._executor = self.sealed = None
            self._binary_scan = None
            self._closed = True
            return
        if self.plan['mode'] == 'historical_parent':
            super().__exit__(kind, value, traceback)
            return
        if isinstance(value, BaseException):
            self._on_native_failure(value)
        try:
            self.close()
        except BaseException as cleanup:
            if not isinstance(value, BaseException):
                raise
            value.add_note('native game inputs cleanup failed: ' + type(cleanup).__name__)
            if getattr(cleanup, '_structured_source_cleanup_owner', None) is self:
                vars(value)['_structured_source_cleanup_owner'] = self
