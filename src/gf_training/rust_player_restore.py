"""Actual current-host DATA resume audit and inference/resume bank owners."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from gf_common.artifacts import ArtifactPhaseAdmission
from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.ram_bank import MAX_BANK_BYTES
from gf_common.rust_player import (
    deadline_seconds,
    validate_exit,
    validate_local_admission,
    validate_operation,
    validate_prepared,
)
from gf_models.structured_sealed import (
    inspect_retained_game_artifact_v3,
    inspect_retained_game_artifact_v4,
)
from gf_training.group_pack_pool import PackedArtifactPool
from gf_training.group_pack_source import OriginalFile
from gf_training.hpo_owned_restore import _ColdScan, _decode_banks_data
from gf_training.lineage import require_hpo_run_owner
from gf_training.rust_player_completion import _audit_retirement, _audit_update, _read
from gf_training.rust_player_lineage import validate_record_shape, validate_run
from gf_training.rust_player_registry import _backings, _PolicyLease, policy_binding_data
from gf_training.rust_player_supervision import HostAdmission, HostControl, read_control_record

_ISSUER = object()


class RustResumeAudit:
    """Previous receipts are DATA under current admission/OFD, never old live caps."""

    state: Any
    current: Any
    artifact: Any

    def __init__(self, control: Any, admission: Any, ownership: Any, *, loaded: Any,
                 pool: Any, writer: Any) -> None:
        if type(control) is not HostControl or type(admission) is not HostAdmission or type(pool) is not PackedArtifactPool:
            raise ContractError('actual CURRENT control/admission/shared pool required')
        admission.require()
        self.ownership = require_hpo_run_owner(ownership.directory, ownership)
        if (admission._control is not control or writer.owner is not ownership or writer.budget is not pool.budget
                or control.operation['execution_request']['mode'] != 'resume'
                or ownership.directory != Path(control.operation['parent_directory'])):
            raise ContractError('actual current resume parent/request/Source owner required')
        self.control, self.admission, self.loaded = control, admission, loaded
        self.pool, self.writer, self.budget = pool, writer, pool.budget
        self.deadline = deadline_seconds(control.operation['deadlines']['execution_bits'])
        self.phase, self._closed, self._failed = 'AUDITING', False, False
        self._loans: list[Any] = []
        self._held: dict[Path, OriginalFile] = {}
        self._partial_banks: list[Any] = []
        self._decodes: list[Any] = []
        self.policies: dict[int, RustRestoredPolicy] = {}
        self.records: dict[int, Json] = {}
        self.record_hashes: dict[int, str] = {}
        self.scan: Any = None
        try:
            self.state, run_sha = _read(ownership.directory / 'run.json', self.budget,
                self.deadline, loans=self._loans)
            hpo = getattr(loaded.config, 'hpo_spec', {}).get('schema_version') == 6
            validate_run(self.state, hpo=hpo)
            actual = admission.observation
            if (self.state['status'] != 'paused' or not self.state['updates']
                    or self.state['config_sha256'] != loaded.sha256
                    or canonical(self.state['effective_config']) != canonical(loaded.effective)
                    or self.state['run_id'] != actual['run_id']
                    or self.state['runtime_binding_sha256'] != actual['runtime_binding_sha256']
                    or self.state['identity']['player_runtime']['producer'] != actual['producer']
                    or self.state['monotonic_authority']['deadlines'] != control.operation['deadlines']
                    or self.state['monotonic_authority']['boot_id'] != control.launch['clock_binding']['boot_id']):
                raise ContractError('actual paused parent/current admission/config/ORIGINAL clock differs')
            selected = []
            for invocation, entry in self.state['invocations'].items():
                path = ownership.directory / f'state-{invocation}.json'
                original = OriginalFile(path, self.deadline)
                if original.sha256 == run_sha:
                    self._held[path] = original
                    selected.append((invocation, entry))
                else:
                    original.close()
            if len(selected) != 1:
                raise ContractError('actual prior immutable paused snapshot is missing/ambiguous')
            invocation, entry = selected[0]
            prior_root = Path(entry['admission_path']).parent
            # B's accepted session bind path preserves one original record root.
            if prior_root != control.root:
                raise ContractError('resume cannot invent a new cross-session allowance')
            operation, operation_sha = read_control_record(Path(entry['operation_path']), prior_root)
            validate_operation(operation, session_finish=control.launch['session_deadlines']['finish_bits'])
            old_admission, admission_sha = read_control_record(Path(entry['admission_path']), prior_root)
            validate_local_admission(old_admission)
            prepared, prepared_sha = read_control_record(prior_root / f'prepared-{invocation}.json', prior_root)
            exit_record, exit_sha = read_control_record(prior_root / f'exit-{invocation}.json', prior_root)
            completion, completion_sha = read_control_record(prior_root / f'completion-{invocation}.json', prior_root)
            validate_prepared(prepared, old_admission)
            validate_exit(exit_record, old_admission)
            if (operation_sha != entry['operation_sha256'] or admission_sha != entry['admission_sha256']
                    or operation['execution_request'] != entry['execution_request']
                    or prepared['state'] != 'paused' or prepared['state_sha256'] != run_sha
                    or exit_record['prepared_sha256'] != prepared_sha or not exit_record['cooperative']
                    or exit_record['wait']['exit_code'] != 0 or not exit_record['descendants_reaped']
                    or exit_record['signals'] or completion['kind'] != 'rust-player-training-completion-v1'
                    or completion['state'] != 'paused' or completion['parent_metadata_validation'] is not None
                    or completion['prepared_sha256'] != prepared_sha or completion['exit_sha256'] != exit_sha
                    or completion['state_sha256'] != run_sha or completion['operation_sha256'] != operation_sha
                    or completion['generation_sha256'] != self.state['generation_sha256']):
                raise ContractError('prior immutable PAUSED operation/Prepared/Exit/completion chain differs')
            self.prior_completion_sha256 = completion_sha
            _audit_retirement(control, self.state, prepared, pool, self._loans, self.deadline, hpo=hpo)
            parent_sha = None
            previous: Any = None
            for generation in range(self.state['updates'] + 1):
                row, checksum = _read(ownership.directory / 'generations' / f'{generation:06}.json',
                    self.budget, self.deadline, loans=self._loans)
                validate_record_shape(row, 'generation', hpo=hpo)
                if row['generation'] != generation or row['parent_sha256'] != parent_sha or row['run_id'] != self.state['run_id']:
                    raise ContractError('actual prior generation chain differs')
                self.records[generation], self.record_hashes[generation] = row, checksum
                if generation:
                    mark = len(self._loans)
                    update, _ = _read(ownership.directory / 'updates' / f'{generation:06}.json',
                        self.budget, self.deadline, row['update_sha256'], loans=self._loans)
                    facts = _audit_update(self, ownership, pool, loaded, self.state, previous, row, update, self.deadline, control_loans=self._loans)
                    self.last_source_manifest_sha256 = facts['manifest_sha256']
                    del update, facts
                    for loan in self._loans[mark:]:
                        loan.close()
                previous, parent_sha = row, checksum
            self.current = self.records[self.state['updates']]
            if parent_sha != prepared['generation_sha256']:
                raise ContractError('actual prior current generation differs from Prepared')
            operating = loaded.config.hpo_spec['operating'] if hpo else {'save_budget_bytes': 512 * 1024**2}
            self.scan = _ColdScan(ownership, ArtifactPhaseAdmission(self.budget, operating['save_budget_bytes']), self.deadline)
            target = ownership.directory / 'artifacts' / f'{self.state["updates"]:06}'
            from gf_common.card_instance_rules import card_instance_rules
            if hpo:
                from gf_models.structured_artifacts import adam_recipe
                recipe = loaded.config.hpo_spec['training_recipe']
                self.artifact = inspect_retained_game_artifact_v4(target,
                    expected_inference_digest=self.current['inference_sha256'], actions=loaded.config.actions,
                    rules=card_instance_rules(), scan=self.scan, training_recipe=recipe,
                    optimizer_recipe=adam_recipe(recipe['adam_learning_rate'], recipe['adam_beta1'], recipe['adam_beta2'], recipe['adam_epsilon']))
            else:
                self.artifact = inspect_retained_game_artifact_v3(target,
                    expected_inference_digest=self.current['inference_sha256'], actions=loaded.config.actions,
                    rules=card_instance_rules(), scan=self.scan)
            assert self.artifact.resume is not None
            if self.artifact.files != self.current['artifact_files'] or self.artifact.resume['resume_identity'] != self.state['resume_identity']:
                raise ContractError('actual full saved pair differs from paused generation')
            self.phase = 'READY'
            self.registry, _ = _read(ownership.directory / 'artifact-registry.json', self.budget,
                self.deadline, loans=self._loans)
            # Register only actual scanned mutable originals; no old write ACK.
            from gf_training.hpo_owned_lineage import _identity
            for name in ('run.json', 'latest.json', 'artifact-registry.json'):
                path = ownership.directory / name
                writer.files[path] = _identity(path)
        except BaseException as error:
            self._failed = True
            self._provisional_owner = vars(error).get('cleanup_owner')
            self._failure_traceback = error.__traceback__
            vars(error)['cleanup_owner'] = self
            raise

    def require(self) -> None:
        self.admission.require()
        require_hpo_run_owner(self.ownership.directory, self.ownership)
        if self._closed or self._failed or self.phase not in ('AUDITING', 'READY', 'LOADED'):
            raise ContractError('actual current live resume DATA audit required')
        for original in self._held.values():
            original.recheck()

    def load(self, local: Any) -> dict[int, RustRestoredPolicy]:
        self.require()
        if self.phase != 'READY' or local.admission is not self.admission:
            raise ContractError('actual same-admission untouched local restore required')
        numbers = set(self.current['parent_controls']['snapshot_pool']) | {self.state['updates']}
        try:
            for generation in sorted(numbers):
                policy = RustRestoredPolicy.__new__(RustRestoredPolicy)
                self.policies[generation] = policy
                RustRestoredPolicy.__init__(policy, _ISSUER, self, local, generation)
            self.phase = 'LOADED'
            return dict(self.policies)
        except BaseException as error:
            self._failed = True
            self._provisional_owner = vars(error).get('cleanup_owner')
            self._failure_traceback = error.__traceback__
            vars(error)['cleanup_owner'] = self
            raise

    def close(self) -> None:
        if self._closed:
            return
        if getattr(self, '_failure_traceback', None) is not None:
            import traceback
            traceback.clear_frames(self._failure_traceback)
        child = getattr(self, '_provisional_owner', None)
        if child is not None and child is not self:
            child.close()
            self._provisional_owner = None
        for policy in self.policies.values():
            policy.close()
        for bank in self._partial_banks:
            bank.close()
        self._partial_banks.clear()
        for receipt in self._decodes:
            if not receipt.closed:
                raise BufferError('actual decoded arrays/tracebacks still own Source admission')
        for original in self._held.values():
            original.close()
        self._held.clear()
        if self.scan is not None:
            self.scan.close()
        self.state = self.current = self.artifact = None
        self.registry = {}
        self.records.clear()
        for loan in self._loans:
            loan.close()
        self._loans.clear()
        self._closed = True

    def __reduce__(self) -> Any:
        raise TypeError('actual current resume audit cannot be copied/serialized')


class RustRestoredPolicy:
    """Held saved files/new readonly banks; original identity is lineage only."""

    def __init__(self, issuer: object, audit: RustResumeAudit, local: Any, generation: int) -> None:
        if issuer is not _ISSUER or type(audit) is not RustResumeAudit:
            raise ContractError('actual completed resume DATA audit required')
        audit.require()
        if audit.phase != 'READY':
            raise ContractError('actual complete original/saved audit required')
        self.audit, self._budget, self._learner = audit, audit.budget, local.learner
        self._consumers: set[Any] = set()
        self._held: dict[str, OriginalFile] = {}
        self._closed = False
        self.inference = self.resume = None
        self.generation = generation
        record = audit.records[generation]
        self.identity = record['snapshot_identity']
        self.generation_sha256 = audit.record_hashes[generation]
        self.optimizer_steps = record['optimizer_steps']
        self.files = dict(record['artifact_files'])
        self.directory = audit.ownership.directory / 'artifacts' / f'{generation:06}'
        self.model, self.binding = local.learner.model, audit.loaded.config.model.inference
        self.actions, self.rules = audit.loaded.config.actions, local.learner._context.rules
        self.manifest = audit.scan.object(self.directory / 'inference.json', record['inference_sha256'])
        self.resume_manifest = audit.scan.object(self.directory / 'resume.json', record['resume_sha256']) if generation == audit.state['updates'] else None
        try:
            for name, checksum in {'inference.json': record['inference_sha256'], **self.manifest['weights']['files'],
                    **({'resume.json': record['resume_sha256'], **self.resume_manifest['state']['files']} if self.resume_manifest else {})}.items():
                original = OriginalFile(self.directory / name, audit.deadline)
                self._held[name] = original
                if original.sha256 != checksum:
                    raise ContractError('actual held restored saved original differs')
            self.inference = _decode_banks_data(self.directory, self.manifest['weights'], model=self.model,
                maximum=MAX_BANK_BYTES, budget=audit.budget, scan=audit.scan,
                partial_banks=audit._partial_banks, decodes=audit._decodes)
            audit._partial_banks.remove(self.inference)
            if self.model.version == '4':
                from gf_models.hpo_owned_inference import _validate_inference_content
                _validate_inference_content(self.inference,
                    {key: value for key, value in self.manifest.items() if key != 'weights'},
                    model=self.model, binding=self.binding, actions=self.actions, rules=self.rules)
            else:
                import numpy as np

                from gf_models.structured_config import parameter_shapes
                expected = parameter_shapes(self.model, layout=self._learner._context.layout)
                if set(self.inference.names) != set(expected):
                    raise ContractError('restored Model3 named inference tensors differ')
                for name, shape in expected.items():
                    array = self.inference.array(name)
                    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
                        raise ContractError('restored Model3 inference tensor content differs')
                    del array
            if self.resume_manifest is not None:
                self.resume = _decode_banks_data(self.directory, self.resume_manifest['state'], model=self.model,
                    maximum=MAX_BANK_BYTES - self.inference.admitted_bytes, budget=audit.budget, scan=audit.scan,
                    partial_banks=audit._partial_banks, decodes=audit._decodes)
                audit._partial_banks.remove(self.resume)
            self._inference_backings = _backings(self.inference)
            self.current_bank_identity = digest({'kind': 'rust-player-restored-backing-v1',
                'original_lineage_identity': self.identity, 'inference_backings': self._inference_backings})
        except BaseException as error:
            vars(error)['cleanup_owner'] = self
            raise

    def require(self) -> None:
        self.audit.require()
        if self._closed or self.inference is None or _backings(self.inference) != self._inference_backings:
            raise ContractError('actual restored inference backing owner differs')
        for original in self._held.values():
            original.recheck()

    def require_restore(self, local: Any, metadata: Json) -> None:
        self.require()
        if (self.resume is None or self._learner is not local.learner
                or local.admission is not self.audit.admission
                or self._budget is not getattr(local, 'budget', self.audit.budget)):
            raise ContractError('actual SAME full restored banks/local/Source required')
        if hashlib.sha256((canonical(metadata) + '\n').encode()).hexdigest() != self._held['resume.json'].sha256:
            raise ContractError('actual restored metadata differs from held resume SHA')

    def policy_binding(self) -> tuple[str, Any]:
        self.require()
        return policy_binding_data(model=self.model, binding=self.binding, directory=self.directory,
            manifest_digest=self.files['inference.json'], actions=self.actions, rules=self.rules)

    def journal_metadata(self) -> Json:
        from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT
        _, binding = self.policy_binding()
        return {'input_contract': EXTERNAL_INPUT_CONTRACT, 'policy_options': binding.policy_options,
            'inference_manifest': self.manifest, 'files': [{'path': name, 'sha256': original.sha256,
                'identity': [getattr(original.identity, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')]}
                for name, original in sorted(self._held.items()) if name == 'inference.json' or name.startswith('weights-')]}

    def retain(self, owner: object) -> _PolicyLease:
        self.require()
        lease = _PolicyLease(self, owner)
        self._consumers.add(lease)
        return lease

    def retire_resume(self) -> None:
        self.require()
        if self.resume is not None:
            self.resume.close()
            self.resume = None

    def close(self) -> None:
        if self._closed:
            return
        if self._consumers:
            raise ContractError('actual restored-policy consumers remain live')
        if self.resume is not None:
            self.resume.close()
        if self.inference is not None:
            self.inference.close()
        for original in self._held.values():
            original.close()
        self._held.clear()
        self.manifest, self.resume_manifest, self.files = {}, None, {}
        self._closed = True

    def __reduce__(self) -> Any:
        raise TypeError('actual restored DATA policy cannot be copied/serialized')
