"""Independent cold original/numeric/saved-state validation; no CUDA or learner."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.rust_player import deadline_seconds
from gf_training.group_pack_pool import PackedArtifactPool
from gf_training.group_pack_resources import close_resources, retain_cleanup
from gf_training.group_pack_source import OriginalFile
from gf_training.lineage import borrow_run_lock
from gf_training.rust_player_lineage import validate_completion, validate_record_shape, validate_run
from gf_training.rust_player_supervision import ColdContinuation, HostControl, write_control_record

_ISSUER = object()
_PARENT_ISSUER = object()


def _audit_retirement(control: HostControl, state: Json, prepared: Json,
                      pool: PackedArtifactPool, loans: list[Any], deadline: float, *, hpo: bool = False,
                      operating: Json | None = None) -> None:
    from gf_training.rust_player_supervision import read_control_record
    close, checksum = read_control_record(Path(prepared['close_ack_path']), control.root)
    if checksum != prepared['close_ack_sha256']:
        raise ContractError('cold close acknowledgement changed')
    invocation = prepared['invocation_id']
    health = state['invocations'][invocation]['host_health']
    if health['errors'] or not all(health[name] for name in ('actors_joined', 'service_joined',
            'writers_joined', 'servers_joined', 'buffers_released', 'gpu_fenced')):
        raise ContractError('cold actual host health is incomplete')
    root = Path(control.operation['parent_directory'])
    bodies = {}
    for name, key, kind in (
        ('native', 'native_retirement_sha256', 'rust-player-native-retirement-v1'),
        ('writers', 'writer_retirement_sha256', 'rust-player-writers-retirement-v1'),
        ('service', 'service_retirement_sha256', 'rust-player-service-retirement-v1'),
        ('cache', 'cache_retirement_sha256', 'rust-player-cache-retirement-v1')):
        if close[key] != health[key]:
            raise ContractError('cold close/health retirement digest differs')
        body, _ = _read(root / 'retirement' / f'{name}-{invocation}.json', pool.budget,
            deadline, close[key], loans=loans)
        if body['schema_version'] != 1 or body['kind'] != kind or body['invocation_id'] != invocation or body['run_id'] != state['run_id']:
            raise ContractError('cold actual retirement record identity differs')
        bodies[name] = body
    if digest(bodies['native']['report']) != bodies['native']['report_sha256']:
        raise ContractError('cold actual native join report differs')
    if (bodies['service']['native_retirement_sha256'] != close['native_retirement_sha256']
            or not bodies['service']['service_joined'] or not bodies['service']['gpu_fenced']
            or bodies['cache']['service_retirement_sha256'] != close['service_retirement_sha256']
            or bodies['cache']['consumer_settlement_sha256'] != bodies['service']['consumer_settlement_sha256']):
        raise ContractError('cold actual service/cache retirement DAG differs')
    registry, _ = _read(root / 'artifact-registry.json', pool.budget, deadline, loans=loans)
    validate_record_shape(registry, 'registry', hpo=hpo)
    if registry['run_id'] != state['run_id'] or registry['runtime_binding_sha256'] != state['runtime_binding_sha256']:
        raise ContractError('cold actual artifact registry identity differs')
    from gf_training.rust_player_retention import audit_artifact_files
    audit_artifact_files(control, state, prepared, registry, pool, loans, deadline, hpo=hpo, operating=operating)
    for token, entry in registry['service_references'].items():
        if entry['status'] != 'released' or entry['release_sha256'] is None:
            raise ContractError('cold cache reference was not actually released')
        directory = root / 'rust-cache-references' / token
        acquisition, _ = _read(directory / 'acquisition.json', pool.budget, deadline,
            entry['acquisition_sha256'], loans=loans)
        publication, _ = _read(directory / 'publication.json', pool.budget, deadline,
            entry['publication_sha256'], loans=loans)
        release, _ = _read(directory / 'release.json', pool.budget, deadline,
            entry['release_sha256'], loans=loans)
        binding = entry['binding']
        if (acquisition['kind'] != 'rust-player-cache-acquisition-v1' or acquisition['binding'] != binding
                or release['kind'] != 'rust-player-cache-release-v1' or release['binding'] != binding
                or release['acquisition_sha256'] != entry['acquisition_sha256']
                or release['publication_sha256'] != entry['publication_sha256']
                or publication['kind'] != 'rust-player-cache-publication-v1'
                or publication['acquisition_sha256'] != entry['acquisition_sha256']
                or publication['source_binding_sha256'] != digest(binding)
                or digest(publication['ack']) != publication['ack_sha256']
                or publication['ack']['source_binding_sha256'] != digest(binding)
                or publication['checkpoint'] != binding['inference_sha256']
                or digest(release['evidence']) != release['evidence_sha256']
                or release['evidence']['entry_absent'] is not True):
            raise ContractError('cold cache acquisition/publication/release correspondence differs')
        if binding['invocation_id'] == invocation and bodies['cache']['releases'].get(token) != entry['release_sha256']:
            raise ContractError('cold cache aggregate omits an actual current invocation release')


class ColdFinalizedRustInvocation:
    """Actual returned CPU DATA owner. It is not a live Source/update admission."""

    completion_path: Path
    completion_sha256: str
    _validated_state: Json | None
    _generation_chain_sha256: str
    _parent_directory: Path
    _provisional_owner: Any

    def __init__(self, issuer: object, cold: ColdContinuation, pool: PackedArtifactPool) -> None:
        if issuer is not _ISSUER:
            raise ContractError('actual independent cold finalizer required')
        self.cold, self.pool = cold, pool
        self._held: list[OriginalFile] = []
        self._workspace: Any = None
        self._control_loans: list[Any] = []
        self.closed = False
        self.artifact: Any = None
        self.completion: Any = None
        self.audit: Any = None
        self._scan: Any
        self._parent_scope = ExitStack()
        self._parent_close_error: BaseException | None = None
        self._failure_traceback: Any = None

    def _hold_parent(self, directory: Path) -> Any:
        self.cold.require_pair()
        self._parent_directory = directory
        self._ownership = self._parent_scope.enter_context(borrow_run_lock(directory, self.cold._control))
        return self._ownership

    def _require_parent(self) -> None:
        from gf_training.lineage import require_hpo_run_owner
        if self.closed:
            raise ContractError('actual returned DATA owner is closed')
        self.cold.require_pair()
        require_hpo_run_owner(self._parent_directory, self._ownership)

    def close(self) -> None:
        if self.closed:
            return
        if hasattr(self, '_ownership'):
            self._require_parent()  # Foreign threads cannot consume the retained context.
        if getattr(self, '_failure_traceback', None) is not None:
            import traceback
            traceback.clear_frames(self._failure_traceback)
        if getattr(self, '_parent_close_error', None) is not None:
            raise ContractError('actual parent borrower close remains uncertain') from self._parent_close_error
        child = getattr(self, '_provisional_owner', None)
        if child is not None:
            if type(child) is ColdFinalizedRustParent and child._invocation is self:
                child._close_metadata()
            else:
                child.close()
        self.artifact = None
        self.audit = self.completion = None
        self._validated_state = None
        if hasattr(self, '_scan'):
            self._scan.files.clear()
            self._scan.inventories.clear()
            self._scan.directory_inventories.clear()
        for original in self._held:
            original.close()
        self._held.clear()
        if hasattr(self, '_parent_scope'):
            try:
                self._parent_scope.close()
            except BaseException as error:
                self._parent_close_error = error
                raise
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None
        for loan in self._control_loans:
            loan.close()
        self._control_loans.clear()
        self.pool.close()
        self.closed = True
        if type(child) is ColdFinalizedRustParent and child._invocation is self:
            child._closed = True
        self._provisional_owner = None

    def __reduce__(self) -> Any:
        raise TypeError('actual cold data ownership cannot be serialized')


class ColdFinalizedRustParent:
    """Exact-D HPO DATA capability, issued only after actual cold output ACK."""

    def __init__(self, issuer: object, invocation: ColdFinalizedRustInvocation) -> None:
        if issuer is not _PARENT_ISSUER or type(invocation) is not ColdFinalizedRustInvocation:
            raise ContractError('actual exact-D cold Parent issuer required')
        state = invocation._validated_state
        completion = invocation.completion
        if (invocation.closed or state is None or completion['parent_metadata_validation'] is not None
                or state['run_variant'] != 'hpo-ppo-owned-rust-player-v1' or state['status'] != 'complete'
                or state['used_decisions'] != state['effective_config']['hpo_spec']['decision_budget']
                or invocation.cold._finalized_sha256 != invocation.completion_sha256
                or not completion['eligible']):
            raise ContractError('actual completed exact-D post-ACK HPO data required')
        self._invocation = invocation
        self._ready, self._failed, self._closed = False, False, False
        self._metadata_original: OriginalFile | None = None
        self._cleanup_children: list[Any] = []
        self._state = canonical(state).encode()
        self._state_sha256 = completion['state_sha256']
        self._generation_sha256 = completion['generation_sha256']
        self._generations_sha256 = invocation._generation_chain_sha256
        self._files = canonical({name: checksum for name, checksum in invocation.artifact.files.items()
            if name == 'inference.json' or name.startswith('weights-')}).encode()

    def require(self) -> None:
        if self._closed or self._failed or not self._ready or self._invocation.closed:
            raise ContractError('actual live qualified exact-D Parent required')
        if self._metadata_original is None:
            raise ContractError('actual held Parent metadata original required')
        self._metadata_original.recheck()
        self._invocation._require_parent()
        for original in self._invocation._held:
            original.recheck()
        if self._invocation.cold._finalized_sha256 != self._invocation.completion_sha256:
            raise ContractError('actual Parent completion identity changed')

    @property
    def state(self) -> Json:
        self.require()
        return json.loads(self._state)

    @property
    def files(self) -> Json:
        self.require()
        return json.loads(self._files)

    def _issue_metadata(self) -> None:
        owner = self._invocation
        state = json.loads(self._state)
        body = {'schema_version': 3, 'kind': 'hpo-rust-parent-metadata-validation',
            'origin': 'producer-owned-v1', 'encoder': 'canonical-python', 'run_id': state['run_id'],
            'run_path': str(owner._parent_directory), 'training_seed': state['seed'],
            'config_sha256': state['config_sha256'], 'state_sha256': self._state_sha256,
            'generation': state['updates'], 'generation_sha256': self._generation_sha256,
            'generations_sha256': self._generations_sha256,
            'inference_sha256': owner.artifact.files['inference.json'], 'used_decisions': state['used_decisions'],
            'effective_settings_sha256': state['effective_config']['hpo_spec']['effective_settings_sha256'],
            'runtime_binding_sha256': state['runtime_binding_sha256'],
            'completion_receipt_sha256': owner.completion_sha256,
            'prepared_sha256': owner.completion['prepared_sha256'], 'exit_sha256': owner.completion['exit_sha256'],
            'source_validation_sha256': owner.completion['audit_sha256']}
        path = owner.cold._control.root / f'parent-metadata-{owner.completion["invocation_id"]}.json'
        try:
            checksum = write_control_record(path, body)
            from gf_training.rust_player_supervision import read_control_record
            actual, actual_sha = read_control_record(path, owner.cold._control.root)
            if actual != body or actual_sha != checksum:
                raise ContractError('actual separate Parent metadata3 changed')
            self._metadata_original = OriginalFile(path, owner.pool.deadline)
            if self._metadata_original.sha256 != checksum:
                raise ContractError('actual held Parent metadata3 changed')
            self.metadata_path, self.metadata_sha256 = path, checksum
            self._metadata = canonical(body).encode()
            self._ready = True
        except BaseException as error:
            self._failed = True
            retain_cleanup(self, error)
            raise

    @property
    def metadata(self) -> Json:
        self.require()
        return json.loads(self._metadata)

    def _close_metadata(self) -> None:
        """Retire this child only; outer invocation owns the enclosing grant."""
        self._ready = False
        self._state = self._files = self._metadata = b''
        if getattr(self, '_cleanup_children', None):
            close_resources(self, [])
        if self._metadata_original is not None:
            self._metadata_original.close()
            self._metadata_original = None

    def close(self) -> None:
        if not self._closed:
            if hasattr(self._invocation, '_ownership'):
                self._invocation._require_parent()
            self._close_metadata()
            self._invocation.close()
            self._closed = True

    def __reduce__(self) -> Any:
        raise TypeError('actual exact-D Parent cannot be copied/serialized')


def _read(path: Path, budget: Any, deadline: float, expected: str | None = None,
          *, loans: list[Any]) -> tuple[Json, str]:
    """Borrowed JSON within this one admitted helper scope, no data authority."""
    original = OriginalFile(path, deadline)
    try:
        if expected is not None and original.sha256 != expected:
            raise ContractError('cold original control checksum differs')
        size = original.identity.st_size
        loans.append(budget.reserve(32 * size + 65536))
        value = json.loads(original.read(size))
        return value, original.sha256
    finally:
        original.close()


class _FailedRawAudit:
    """One stopped raw audit's actual Originals and funded traceback aliases."""

    def __init__(self, originals: Any, error: BaseException) -> None:
        self.originals, self.traceback = originals, error.__traceback__
        self.nested = vars(error).get('cleanup_owner')
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        import traceback
        traceback.clear_frames(self.traceback)
        if self.nested is not None and self.nested is not self.originals:
            self.nested.close()
        self.originals.close()
        self.traceback, self.nested, self.closed = None, None, True


def _audit_raw_originals(originals: Any, *, loaded: Any, state: Json, previous: Json,
                         update: Json, proof: Json, generation: int, hpo: bool,
                         expected: Path) -> tuple[int, int, tuple[str, ...], str]:
    """Private raw DATA audit; live-owner construction remains in _audit_update."""
    from gf_common.card_instance_collection import PPO_BATCH, hpo_batch_role
    from gf_training.card_instance_collection_source import inspect_rust_game_manifest
    from gf_training.structured_rollout import _episode_correspondence_digest
    from gf_training.structured_source import _inspect_collection_content
    try:
        manifest = originals.object(originals.root / 'manifest.json')
        assert originals.source is not None
        actual_storage = {'format': 'sqlite-group-pack-v1', 'descriptor_path': str(expected),
            'descriptor_sha256': originals._descriptor_sha256,
            'database_sha256': originals.source._originals.descriptor['database_sha256'],
            'seal_id': originals.source._originals.descriptor['seal_id'],
            'logical_sha256': originals.source._originals.validated['seal']['logical_sha256']}
        if (digest(manifest) != update['collection_manifest_sha256']
                or digest(manifest) != proof['manifest_sha256']
                or canonical(manifest['config']) != canonical(loaded.effective)
                or canonical(manifest['code']) != canonical(state['identity']['code'])
                or manifest['config_hash'] != loaded.sha256
                or manifest['runtime_binding_sha256'] != state['runtime_binding_sha256']
                or manifest['player_producer'] != proof['player_producer']
                or digest(manifest['capture_authority']) != proof['capture_authority_sha256']
                or digest(manifest['logical_retirement']) != proof['collection_retirement_sha256']
                or canonical(proof['storage']) != canonical(actual_storage)):
            raise ContractError('cold actual originals differ from frozen Source/runtime/config')
        role = hpo_batch_role(loaded.config.hpo_spec['training_recipe']['collection_games']) if hpo else PPO_BATCH
        episodes, specs = inspect_rust_game_manifest(originals, manifest, loaded.config.actions, 'train', role=role)
        outputs, accepted, closures, missing = _inspect_collection_content(originals,
            loaded.config.actions, manifest, episodes, specs, _game=True, packed=True)
        seats = update['learner_seats']
        records = tuple(row.records for row in outputs if row.viewer == seats[row.episode_id])
        episode_digests = tuple(_episode_correspondence_digest(record) for record in records)
        collected = sum(len(record.frames) for record in records)
        count = collected
        if hpo:
            from gf_training.hpo_collection import selection_for_counts
            selection = selection_for_counts(tuple(len(record.frames) for record in records),
                learning_seed=state['seed'], batch=generation - 1,
                decision_budget=loaded.config.hpo_spec['decision_budget'], used_before=previous['used_decisions'])
            count = selection['used_decisions']
            if (canonical(selection) != canonical(proof['selection'])
                    or digest(selection) != proof['selection_sha256']
                    or update['used_decisions'] != count or update['collected_decisions'] != collected
                    or proof['replay_reconciled'] is not True):
                raise ContractError('cold independently replayed HPO endpoint selection differs')
        if (canonical(episode_digests) != canonical(proof['episode_digests'])
                or count != update['accepted_decisions'] or accepted != proof['accepted_decisions']
                or closures != {'natural': role.games} or missing
                or canonical(originals.files) != canonical(proof['files'])
                or digest(originals.files) != proof['source_set_sha256']):
            raise ContractError('cold FULL raw/Replay/count/file correspondence differs')
        correspondence = {'recipe': 'rust-player-source-numeric-v1',
            'manifest_sha256': proof['manifest_sha256'], 'episode_digests': episode_digests,
            'snapshot_identity': previous['snapshot_identity'], 'numeric_files': update['numeric_files']}
        if digest(correspondence) != proof['source_correspondence']:
            raise ContractError('cold original/numeric correspondence differs')
        source_invocation = manifest['capture_authority']['invocation_id']
        if (source_invocation != manifest['logical_retirement']['invocation_id']
                or source_invocation != manifest['service_usage']['invocation_id']):
            raise ContractError('actual Source invocation witnesses differ')
        # Do not retain a second whole raw graph while reconstructing numeric.
        del outputs, records, manifest, episodes, specs
        originals.close()
    except BaseException as error:
        vars(error)['cleanup_owner'] = _FailedRawAudit(originals, error)
        raise
    return count, collected, episode_digests, source_invocation


def _audit_source_device_plan(update: Json, *, loaded: Any, state: Json, source_invocation: str,
                              record_root: Path, pool: PackedArtifactPool, deadline: float,
                              loans: list[Any]) -> None:
    """Saved source receipts are DATA, not an old live HostAdmission/operation."""
    from dataclasses import replace

    from gf_common.rust_player import validate_local_admission, validate_operation
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.rust_player_device import CudaCheckpointProfile, bind_update_device_plan
    from gf_models.structured_config import resource_estimate
    learner, serving = loaded.config.training.device, loaded.config.model.inference.device
    if learner == serving == 'cpu':
        bind_update_device_plan(update, operation_id='', learner_device=learner,
            serving_device=serving, global_bytes=0)
        return
    entry = state['invocations'][source_invocation]
    op_path, admit_path = Path(entry['operation_path']), Path(entry['admission_path'])
    if any(not path.is_absolute() or not path.is_relative_to(record_root) or '..' in path.parts
            for path in (op_path, admit_path)):
        raise ContractError('source operation/admission leaves original record root')
    operation, operation_sha = _read(op_path, pool.budget, deadline, entry['operation_sha256'], loans=loans)
    admission, admission_sha = _read(admit_path, pool.budget, deadline, entry['admission_sha256'], loans=loans)
    validate_operation(operation)
    validate_local_admission(admission)
    if (operation['schema_version'] != 2 or admission['invocation_id'] != source_invocation
            or operation_sha != entry['operation_sha256'] or admission_sha != entry['admission_sha256']
            or operation['operation_id'] != admission['operation_id']
            or operation['run_id'] != admission['run_id'] or operation['run_id'] != state['run_id']
            or operation['config_sha256'] != admission['config_sha256'] or operation['config_sha256'] != loaded.sha256
            or operation['runtime_binding_sha256'] != admission['runtime_binding_sha256']
            or operation['runtime_binding_sha256'] != state['runtime_binding_sha256']
            or admission['producer'] != state['identity']['player_runtime']['producer']
            or operation['deadlines'] != admission['deadlines'] or operation['expected_context'] != admission['context']):
        raise ContractError('embedded plan actual source operation/admission binding differs')
    plan = bind_update_device_plan(update, operation_id=operation['operation_id'], learner_device=learner,
        serving_device=serving, global_bytes=admission['budget']['managed_cuda_bytes'])
    assert plan is not None
    model = loaded.config.model.architecture(layout=layout)
    binding = replace(loaded.config.model.inference, device=learner)
    profile = CudaCheckpointProfile(model, binding, layout=layout)
    hpo = getattr(loaded.config, 'hpo_spec', {}).get('schema_version') == 6
    logical = (resource_estimate(model, binding,
        suffix_decisions=loaded.config.hpo_spec['training_recipe']['tbptt_length'], layout=layout)['total_bytes']
        if hpo else min(model.resources.max_owned_bytes, binding.managed_budget_bytes))
    expected = {'logical_bytes': logical, 'host_fixed_bytes': profile.fixed_bytes,
        'learner_fixed_bytes': profile.fixed_bytes if learner != 'cpu' else 0,
        'post_update_learner_tensor_bytes': 4 * profile.parameter_bytes + profile.position_bytes if learner != 'cpu' else 0,
        'publication_network_bytes': profile.network_bytes if serving != 'cpu' else 0}
    if any(plan[name] != value for name, value in expected.items()):
        raise ContractError('device plan differs from actual self-pool model/resource recipe')


def _audit_update(cold: Any, ownership: Any, pool: PackedArtifactPool,
                  loaded: Any, state: Json, previous: Json, current: Json,
                  update: Json, deadline: float, *, control_loans: list[Any]) -> Json:
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.group_pack_game import storage_binding
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.structured_artifacts import semantics_v3
    from gf_models.structured_sealed import basename
    from gf_training.rust_player_source import RustColdOriginals, RustResumeOriginals
    from gf_training.structured_rollout import (
        _source_correspondence,
        read_game_rollout,
    )

    hpo = getattr(loaded.config, 'hpo_spec', {}).get('schema_version') == 6
    validate_record_shape(update, 'update', hpo=hpo)
    proof = validate_record_shape(update['source_proof'], 'proof', hpo=hpo)
    generation = current['generation']
    expected = Path(loaded.config.storage.db_root) / state['run_id'] / 'collections' / f'batch-{generation - 1:06}' / 'collection.pending.json'
    if (proof['storage']['descriptor_path'] != str(expected) or update['generation'] != generation
            or update['parent_sha256'] != current['parent_sha256']
            or update['parent_snapshot_identity'] != previous['snapshot_identity']
            or update['result_snapshot_identity'] != current['snapshot_identity']
            or proof['snapshot_identity'] != previous['snapshot_identity']
            or proof['runtime_binding_sha256'] != state['runtime_binding_sha256']
            or proof['player_producer'] != state['identity']['player_runtime']['producer']):
        raise ContractError('cold Source/update/generation ownership correspondence differs')
    if type(cold) is ColdContinuation:
        originals: Any = RustColdOriginals(cold, ownership, expected, proof['storage']['descriptor_sha256'],
            pool=pool, settings=loaded.config.preprocessing, deadline=deadline)
    else:
        from gf_training.rust_player_restore import RustResumeAudit
        if type(cold) is not RustResumeAudit:
            raise ContractError('actual cold or current-host resume audit required')
        originals = RustResumeOriginals(cold, expected, proof['storage']['descriptor_sha256'],
            settings=loaded.config.preprocessing, deadline=deadline)
    count, _collected, episode_digests, source_invocation = _audit_raw_originals(originals, loaded=loaded,
        state=state, previous=previous, update=update, proof=proof,
        generation=generation, hpo=hpo, expected=expected)
    mark = len(control_loans)
    control = cold._control if type(cold) is ColdContinuation else cold.control
    _audit_source_device_plan(update, loaded=loaded, state=state, source_invocation=source_invocation,
        record_root=control.root, pool=pool, deadline=deadline, loans=control_loans)
    # The helper returned no borrowed JSON. Failure instead leaves loans with
    # the existing outer owner until its stopped-frame/child cleanup succeeds.
    for loan in control_loans[mark:]:
        loan.close()
    del control_loans[mark:]
    model = loaded.config.model.architecture(layout=layout)
    if hpo:
        from gf_training.structured_rollout import _hpo_semantics, _RolloutContext
        declaration = _hpo_semantics(model, _RolloutContext(loaded.config.actions, card_instance_rules()))
    else:
        declaration = semantics_v3(model, layout=layout,
            action_contract=loaded.config.actions.action_contract, rules=card_instance_rules())
    numeric = ownership.directory / update['numeric']
    if numeric != ownership.directory / 'numeric' / f'batch-{generation - 1:06}':
        raise ContractError('cold numeric directory differs from exact batch')
    for name, checksum in update['numeric_files'].items():
        basename(name)
        original = OriginalFile(numeric / name, deadline)
        try:
            if original.sha256 != checksum:
                raise ContractError('cold numeric original checksum differs')
        finally:
            original.close()
    reservation = pool.budget.reserve(0)
    try:
        readback: Any
        correspondence = _source_correspondence(digest(proof), declaration, episode_digests,
            game=True, hpo=hpo, storage_binding=storage_binding())
        if hpo:
            from gf_training.hpo_config import hpo_training_config
            from gf_training.hpo_rollout import read_hpo_rollout
            readback = read_hpo_rollout(numeric, update['numeric_sha256'], model,
                hpo_training_config(loaded.config.hpo_spec['training_recipe']), actions=loaded.config.actions,
                rules=card_instance_rules(), expected_selection=proof['selection'],
                expected_correspondence=correspondence, expected_snapshot_identity=proof['snapshot_identity'],
                storage_binding=storage_binding(), numeric_budget_bytes=loaded.config.hpo_spec['operating']['numeric_budget_bytes'],
                max_owned_bytes=pool.budget.limit, reserve=reservation.resize)
        else:
            readback = read_game_rollout(numeric, update['numeric_sha256'], model,
                actions=loaded.config.actions, rules=card_instance_rules(), declaration=declaration,
                expected_correspondence=correspondence, storage_binding=storage_binding(),
                max_owned_bytes=pool.budget.limit, reserve=reservation.resize)
        readback.require_learning()
        if len(readback) != count:
            raise ContractError('cold numeric learner count differs')
        del readback
    finally:
        reservation.close()
    return {'generation': generation, 'manifest_sha256': proof['manifest_sha256'],
        'numeric_sha256': update['numeric_sha256'], 'learner_decisions': count,
        'episode_digests': episode_digests, 'source_set_sha256': proof['source_set_sha256']}


def _audit_empty_schedule(manifest: Json, *, state: Json, current: Json, loaded: Any,
                          parent: Path, inference_files: dict[int, str]) -> None:
    """Pure schedule correspondence from saved RNG/pool and retained identities."""
    import numpy as np

    from gf_common.schedule import serialize_schedule
    from gf_common.training_opponents import training_opponents
    from gf_training.hpo_collection import training_groups_hpo, training_groups_hpo_refresh
    labels = {}
    for label, binding in manifest['policy_bindings'].items():
        if binding['model_id'] != 'mlp':
            continue
        options = binding['policy_options']
        path = Path(options['checkpoint'])
        matches = [generation for generation, sha in inference_files.items()
            if path == parent / 'artifacts' / f'{generation:06}' and options['expected_manifest_digest'] == sha]
        if len(matches) != 1 or matches[0] in labels:
            raise ContractError('empty collection inference binding is not the frozen pool/current generation')
        labels[matches[0]] = label
    if set(labels) != set(inference_files):
        raise ContractError('empty collection omits a frozen inference pool/current binding')
    learner = labels[state['updates']]
    snapshots = tuple(labels[g] for g in current['parent_controls']['snapshot_pool'])
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = current['parent_controls']['schedule_opponent_rng']
    games = loaded.config.hpo_spec['training_recipe']['collection_games']
    mixed = training_opponents(loaded.effective)
    groups = (training_groups_hpo_refresh(state['seed'], state['updates'], learner, snapshots, rng,
        collection_games=games, actions=loaded.config.actions, training_opponents=mixed) if mixed is not None else
        training_groups_hpo(state['seed'], state['updates'], learner, snapshots, rng, collection_games=games))
    if canonical(serialize_schedule(groups, 4)) != canonical(manifest['schedule']):
        raise ContractError('empty collection differs from independently reconstructed frozen schedule')


def _audit_empty_originals(originals: Any, *, loaded: Any, state: Json, current: Json,
                           marker: Json, admission: Json, parent: Path, inference_files: dict[int, str]) -> None:
    """Extra sealed Raw/Replay audit with no numeric reader or update proof."""
    from gf_common.card_instance_collection import hpo_batch_role
    from gf_common.domain import closed
    from gf_training.card_instance_collection_source import inspect_rust_game_manifest
    from gf_training.rust_player_source import learner_records
    from gf_training.structured_source import _inspect_collection_content
    try:
        manifest = originals.object(originals.root / 'manifest.json')
        link = manifest['linkage']
        capture, retirement = manifest['capture_authority'], manifest['logical_retirement']
        closed(capture, {'schema_version', 'kind', 'actor_protocol_version', 'run_id', 'batch_id',
            'generation', 'invocation_id', 'admission_sha256', 'runtime_binding_sha256',
            'encoding_contexts_sha256', 'accepted_records', 'completed_reports_sha256'})
        closed(retirement, {'schema_version', 'kind', 'run_id', 'batch_id', 'generation', 'invocation_id',
            'admission_sha256', 'runtime_binding_sha256', 'service_id', 'episode_ids', 'completed_reports',
            'accepted_records_sha256', 'service_settlement', 'service_settlement_sha256', 'gpu_fenced'})
        if (digest(manifest) != marker['manifest_sha256'] or originals._descriptor_sha256 != marker['descriptor_sha256']
                or originals.root / 'manifest.json' != Path(marker['manifest_path'])
                or canonical(manifest['config']) != canonical(loaded.effective)
                or canonical(manifest['code']) != canonical(state['identity']['code'])
                or canonical(manifest['inputs']) != canonical(state['config_inputs'])
                or manifest['config_hash'] != loaded.sha256
                or manifest['runtime_binding_sha256'] != state['runtime_binding_sha256']
                or manifest['player_producer'] != admission['producer']
                or manifest['snapshot_identity'] != current['snapshot_identity']
                or link['parent_run_id'] != state['run_id'] or link['parent_run_path'] != str(parent)
                or link['collection_generation'] != state['updates'] or link['batch_id'] != marker['batch_id']):
            raise ContractError('empty original manifest/config/current parent differs')
        for row in (capture, retirement):
            if (type(row['schema_version']) is not int or row['schema_version'] != 1
                    or row['run_id'] != state['run_id'] or row['batch_id'] != marker['batch_id']
                    or row['generation'] != state['updates'] or row['invocation_id'] != admission['invocation_id']
                    or row['admission_sha256'] != digest(admission)
                    or row['runtime_binding_sha256'] != state['runtime_binding_sha256']):
                raise ContractError('empty capture/retirement actual admission differs')
        if (capture['kind'] != 'rust-player-capture-authority-v1' or capture['actor_protocol_version'] != 11
                or retirement['kind'] != 'rust-player-collection-retirement-v1' or retirement['gpu_fenced'] is not True
                or capture['completed_reports_sha256'] != digest(retirement['completed_reports'])
                or retirement['accepted_records_sha256'] != digest(capture['accepted_records'])
                or retirement['service_settlement_sha256'] != digest(retirement['service_settlement'])):
            raise ContractError('empty actual capture/retirement correspondence differs')
        settlement = retirement['service_settlement']
        if (settlement['run_id'] != state['run_id'] or settlement['batch_id'] != marker['batch_id']
                or settlement['generation'] != state['updates'] or settlement['invocation_id'] != admission['invocation_id']
                or settlement['service_id'] != retirement['service_id'] or settlement['gpu_fenced'] is not True
                or any(type(settlement[key]) is not int or settlement[key] != 0 for key in
                    ('active_policy_references', 'native_pending', 'started_producers'))):
            raise ContractError('empty actual consumer settlement differs')
        from gf_common.hpo import selection_context
        expected_context = selection_context(decision_budget=loaded.config.hpo_spec['decision_budget'],
            used_before=state['used_decisions'], learning_seed=state['seed'], batch=state['updates'])
        if canonical(link['selection_context']) != canonical(expected_context):
            raise ContractError('empty collection frozen selection context differs')
        _audit_empty_schedule(manifest, state=state, current=current, loaded=loaded, parent=parent, inference_files=inference_files)
        role = hpo_batch_role(loaded.config.hpo_spec['training_recipe']['collection_games'])
        episodes, specs = inspect_rust_game_manifest(originals, manifest, loaded.config.actions, 'train', role=role)
        outputs, _accepted, closures, missing = _inspect_collection_content(originals,
            loaded.config.actions, manifest, episodes, specs, _game=True, packed=True)
        records = learner_records(outputs, link['learner_seats'], closures, missing, games=role.games)
        from gf_training.hpo_collection import selection_for_counts
        selection = selection_for_counts(tuple(len(record.frames) for record in records),
            learning_seed=state['seed'], batch=state['updates'], decision_budget=loaded.config.hpo_spec['decision_budget'],
            used_before=state['used_decisions'])
        if sum(len(record.frames) for record in records) or selection['used_decisions']:
            raise ContractError('empty marker contradicts independently replayed learner decisions')
        del outputs, records, manifest, episodes, specs, link, capture, retirement, settlement
        originals.close()
    except BaseException as error:
        failure = _FailedRawAudit(originals, error)
        vars(error)['cleanup_owner'] = failure
        raise


def _audit_empty(cold: ColdContinuation, ownership: Any, pool: PackedArtifactPool,
                 loaded: Any, state: Json, current: Json, marker: Json, deadline: float, loans: list[Any]) -> None:
    from gf_training.rust_player_lineage import validate_empty_collection
    from gf_training.rust_player_source import RustColdOriginals
    validate_empty_collection(marker, state=state, config=loaded.config)
    admission, _, _ = cold.require_pair()
    originals = RustColdOriginals(cold, ownership, Path(marker['descriptor_path']), marker['descriptor_sha256'],
        pool=pool, settings=loaded.config.preprocessing, deadline=deadline)
    mark = len(loans)
    inference_files = {state['updates']: current['inference_sha256']}
    try:
        if (ownership.directory / 'numeric' / marker['batch_id']).exists():
            raise ContractError('empty batch must not have a numeric payload')
        for generation in current['parent_controls']['snapshot_pool']:
            if generation not in inference_files:
                row, _ = _read(ownership.directory / 'generations' / f'{generation:06}.json', pool.budget,
                    deadline, loans=loans)
                inference_files[generation] = row['inference_sha256']
                del row
    except BaseException as error:
        vars(error)['cleanup_owner'] = _FailedRawAudit(originals, error)
        raise
    _audit_empty_originals(originals, loaded=loaded, state=state, current=current,
        marker=marker, admission=admission, parent=ownership.directory, inference_files=inference_files)
    inference_files.clear()
    for loan in loans[mark:]:
        loan.close()
    del loans[mark:]


def finalize_cold(control: HostControl, cold: ColdContinuation, *, loaded: Any) -> ColdFinalizedRustInvocation | ColdFinalizedRustParent:
    """Actual fixed cold entry; full ordinary data audit, no Torch/CUDA calls."""
    import hashlib
    from types import SimpleNamespace

    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.structured_sealed import (
        SealedScan,
        inspect_retained_game_artifact_v3,
        inspect_retained_game_artifact_v4,
    )
    from gf_models.torch_runtime import runtime_metadata
    from gf_training.rust_player_supervision import runtime_producer

    if type(control) is not HostControl or type(cold) is not ColdContinuation or cold is not control.cold:
        raise ContractError('actual original cold continuation required')
    admission, prepared, _exit_record = cold.require_pair()
    hpo = getattr(loaded.config, 'hpo_spec', {}).get('schema_version') == 6
    if control.operation['schema_version'] != 2 or control.operation['operation_type'] != ('hpo-train' if hpo else 'ordinary-train'):
        raise ContractError('actual known operation2 cold entry required')
    metadata = runtime_metadata()
    sampler = digest({'format': 'cpu-torch-generator-v1', 'runtime': metadata,
        'module_cuda': '13.2' if metadata['torch'].endswith('+cu132') else None})
    supervisor = Path(f'/proc/{control._parent_pid}/exe')
    producer = runtime_producer(SimpleNamespace(manifest={
        'executable_sha256': hashlib.sha256(supervisor.read_bytes()).hexdigest()}), sampling_runtime_sha256=sampler)
    if producer != admission['producer']:
        raise ContractError('cold actual Python/native/supervisor/sampling producer differs')
    deadline = deadline_seconds(admission['deadlines']['finish_bits'])
    parent = Path(control.operation['parent_directory'])
    pool = PackedArtifactPool(max_owned_bytes=loaded.config.preprocessing.memory_budget_mib * 1024**2, deadline=deadline)
    result = ColdFinalizedRustInvocation(_ISSUER, cold, pool)
    try:
        ownership = result._hold_parent(parent)
        state, state_sha = _read(parent / f'state-{admission["invocation_id"]}.json', pool.budget,
            deadline, prepared['state_sha256'], loans=result._control_loans)
        validate_run(state, hpo=hpo)
        invocation = state['invocations'][admission['invocation_id']]
        if (state_sha != prepared['state_sha256'] or state['run_id'] != admission['run_id']
                or state['config_sha256'] != loaded.sha256
                or canonical(state['effective_config']) != canonical(loaded.effective)
                or state['host_result'] != 'awaiting-supervised-finalization'
                or invocation['operation_sha256'] != control.operation_sha256
                or invocation['operation_path'] != str(control.operation_path)
                or invocation['execution_request'] != control.operation['execution_request']
                or invocation['admission_sha256'] != digest(admission)
                or state['generation_sha256'] != prepared['generation_sha256']):
            raise ContractError('cold frozen state/current actual operation/Prepared differs')
        _audit_retirement(control, state, prepared, pool, result._control_loans, deadline, hpo=hpo,
            operating=loaded.config.hpo_spec['operating'] if hpo else None)
        updates: list[Json] = []
        previous: Any = None
        parent_sha: Any = None
        previous_loans: list[Any] = []
        current = None
        chain_digest = hashlib.sha256()
        chain_digest.update(b'[')
        for generation in range(state['updates'] + 1):
            mark = len(result._control_loans)
            current, checksum = _read(parent / 'generations' / f'{generation:06}.json', pool.budget,
                deadline, loans=result._control_loans)
            current_loans = result._control_loans[mark:]
            validate_record_shape(current, 'generation', hpo=hpo)
            if generation:
                chain_digest.update(b',')
            chain_digest.update(canonical(current).encode())
            if current['generation'] != generation or current['parent_sha256'] != parent_sha or current['run_id'] != state['run_id']:
                raise ContractError('cold actual generation chain differs')
            if generation:
                path = parent / 'updates' / f'{generation:06}.json'
                # The decoded update/proof graph remains funded for its full
                # raw/numeric audit; return only compact facts afterward.
                mark = len(result._control_loans)
                update, _ = _read(path, pool.budget, deadline, current['update_sha256'], loans=result._control_loans)
                update_loans = result._control_loans[mark:]
                facts = _audit_update(cold, ownership, pool, loaded, state,
                previous, current, update, deadline, control_loans=result._control_loans)
                updates.append(facts)
                del update
                from gf_training.structured_rollout import _owned_bytes
                update_loans[0].resize(_owned_bytes(facts))
            previous, parent_sha = current, checksum
            for loan in previous_loans:
                loan.close()
            previous_loans = current_loans
        chain_digest.update(b']')
        marker = invocation.get('empty_collection')
        if hpo and ((state['stop_reason'] == 'empty_collection') != (marker is not None)):
            raise ContractError('empty marker must belong to the actual current frozen invocation')
        if marker is not None:
            if current is None:
                raise ContractError('empty collection requires genuine current saved generation')
            _audit_empty(cold, ownership, pool, loaded, state, current, marker, deadline, result._control_loans)
        expected_batches = {f'batch-{number:06}' for number in range(state['updates'] + int(marker is not None))}
        collection_root = Path(loaded.config.storage.db_root) / state['run_id'] / 'collections'
        if collection_root.exists() and {path.name for path in collection_root.iterdir()} != expected_batches:
            raise ContractError('cold unexpected committed/uncommitted collection inventory')
        last_source = marker['manifest_sha256'] if marker is not None else (updates[-1]['manifest_sha256'] if updates else None)
        if (parent_sha != prepared['generation_sha256'] or current is None
                or digest(current['artifact_files']) != prepared['artifacts_sha256']
                or last_source != prepared['source_manifest_sha256']):
            raise ContractError('cold current committed artifacts/source differs')
        # Existing sealed content scanner owns no CUDA API. Its declared
        # remaining workspace is the SAME actual pool grant through return.
        grant = pool.budget.limit - pool.budget.used
        result._workspace = pool.budget.reserve(grant)
        scan = SealedScan(max_owned_bytes=grant, deadline=deadline)
        result._scan = scan
        target = parent / 'artifacts' / f'{state["updates"]:06}'
        if hpo:
            from gf_models.structured_artifacts import adam_recipe
            recipe = loaded.config.hpo_spec['training_recipe']
            artifact = inspect_retained_game_artifact_v4(target,
                expected_inference_digest=current['inference_sha256'], actions=loaded.config.actions,
                rules=card_instance_rules(), scan=scan, training_recipe=recipe,
                optimizer_recipe=adam_recipe(recipe['adam_learning_rate'], recipe['adam_beta1'],
                    recipe['adam_beta2'], recipe['adam_epsilon']))
        else:
            artifact = inspect_retained_game_artifact_v3(target,
                expected_inference_digest=current['inference_sha256'], actions=loaded.config.actions,
                rules=card_instance_rules(), scan=scan)
        assert artifact.resume is not None
        if (artifact.files != current['artifact_files'] or artifact.resume['resume_identity'] != state['resume_identity']
                or artifact.resume['optimizer_steps'] != state['optimizer_steps']
                or current['optimizer_steps'] != state['optimizer_steps']):
            raise ContractError('cold FULL saved tensor/state/generation content differs')
        for name, checksum in artifact.files.items():
            original = OriginalFile(target / name, deadline)
            result._held.append(original)
            if original.sha256 != checksum:
                raise ContractError('cold held saved original changed')
        result.artifact = artifact
        audit = {'schema_version': 1, 'kind': 'rust-player-cold-audit-v1', 'run_id': state['run_id'],
            'invocation_id': admission['invocation_id'], 'operation_sha256': control.operation_sha256,
            'prepared_sha256': cold._payload['prepared_sha256'], 'exit_sha256': cold._payload['exit_sha256'],
            'state_sha256': state_sha, 'generation_sha256': parent_sha, 'updates': updates,
            'saved_pair': {'generation': state['updates'], 'inference_sha256': current['inference_sha256'],
                'resume_sha256': current['resume_sha256'], 'artifact_files': artifact.files,
                'optimizer_steps': state['optimizer_steps'], 'full_content_validated': True},
            'runtime_binding_sha256': state['runtime_binding_sha256']}
        audit_path = control.root / f'cold-audit-{admission["invocation_id"]}.json'
        audit_sha = write_control_record(audit_path, audit)
        completion = {'schema_version': 1, 'kind': 'rust-player-training-completion-v1',
            **{key: admission[key] for key in ('supervision_id', 'invocation_id', 'operation_id',
                'run_id', 'config_sha256', 'runtime_binding_sha256')},
            'operation_sha256': control.operation_sha256, 'prepared_sha256': cold._payload['prepared_sha256'],
            'exit_sha256': cold._payload['exit_sha256'], 'state_sha256': state_sha,
            'generation_sha256': parent_sha, 'source_manifest_sha256': prepared['source_manifest_sha256'],
            'state': state['status'], 'committed_update': state['updates'], 'audit_path': str(audit_path),
            'audit_sha256': audit_sha, 'eligible': state['status'] == 'complete' and (
                not hpo or state['used_decisions'] == loaded.config.hpo_spec['decision_budget']),
            'parent_metadata_validation': None}
        validate_completion(completion)
        completion_path = control.root / f'completion-{admission["invocation_id"]}.json'
        import time

        from gf_simulator_client.performance_diagnostics import host_diagnostics
        diagnostics = host_diagnostics()
        written_completion_sha = write_control_record(completion_path, completion)
        written_ns = time.monotonic_ns() if diagnostics is not None else None
        if diagnostics is not None:
            diagnostics.phase('completion_written', diagnostics.scope(run_id=admission['run_id'],
                operation_id=admission['operation_id'], invocation_id=admission['invocation_id'], phase='cold'), monotonic_ns=written_ns,
                links={'operation_sha256': control.operation_sha256,
                    'prepared_sha256': cold._payload['prepared_sha256'], 'exit_sha256': cold._payload['exit_sha256'],
                    'completion_sha256': written_completion_sha})
        completion_sha = cold.acknowledge_finalizer_output(completion_path)
        result.completion, result.audit = completion, audit
        result.completion_path = completion_path
        result.completion_sha256 = completion_sha
        result._validated_state = state
        result._generation_chain_sha256 = chain_digest.hexdigest()
        result._parent_directory = parent
        if hpo and completion['eligible']:
            qualified = ColdFinalizedRustParent(_PARENT_ISSUER, result)
            result._provisional_owner = qualified
            qualified._issue_metadata()
            return qualified
        return result
    except BaseException as error:
        if vars(error).get('cleanup_owner') is not None and vars(error)['cleanup_owner'] is not result:
            result._provisional_owner = vars(error)['cleanup_owner']
        vars(error)['cleanup_owner'] = result
        result._failure_traceback = error.__traceback__
        raise
