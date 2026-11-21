"""Rust artifact policy and original-handle retirement; JSON is data only."""
from __future__ import annotations

import hashlib
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer
from gf_common.rust_player import hex_identity
from gf_training.group_pack_resources import OwnedDescriptor, close_resources, retain_cleanup

_COMMON = {'schema_version', 'kind', 'run_id', 'operation_id', 'invocation_id', 'retirement_id',
    'generation', 'file', 'sha256', 'original_identity', 'runtime_binding_sha256'}


def validate_retirement_record(value: Json, *, receipt: bool) -> Json:
    extra = {'intent_sha256', 'completed_unix'} if receipt else {'staging_basename',
        'parent_identity', 'source_directory_identity', 'stage_directory_identity'}
    row = closed(value, _COMMON | extra)
    kind = 'receipt' if receipt else 'intent'
    if type(row['schema_version']) is not int or row['schema_version'] != 1 or row['kind'] != f'rust-player-artifact-retirement-{kind}-v1':
        raise ContractError('explicit Rust artifact retirement record required')
    if type(row['run_id']) is not str or not row['run_id']:
        raise ContractError('retirement run identity required')
    if type(row['operation_id']) is not str or not row['operation_id']:
        raise ContractError('actual operation identity required')
    for name in ('invocation_id', 'retirement_id'):
        hex_identity(row[name], 32)
    for name in ('sha256', 'runtime_binding_sha256', *(['intent_sha256'] if receipt else [])):
        hex_identity(row[name])
    integer(row['generation'], 0)
    from gf_models.structured_sealed import basename
    basename(row['file'])
    identities = {'original_identity': 5} if receipt else {'original_identity': 5,
        'parent_identity': 2, 'source_directory_identity': 2, 'stage_directory_identity': 2}
    for name, length in identities.items():
        if type(row[name]) is not list or len(row[name]) != length:
            raise ContractError('retirement original identity shape differs')
        for number in row[name]:
            integer(number, 0)
    if receipt:
        import math
        completed = row['completed_unix']
        if type(completed) not in (int, float) or not math.isfinite(completed) or completed < 0:
            raise ContractError('finite nonnegative retirement time required')
    elif row['staging_basename'] != '.prune-' + row['retirement_id']:
        raise ContractError('retirement staging identity differs')
    return row


def _ordinary_protected_generations(registry: Json, generation: int, snapshot_pool: Any,
                                    pending: list[int] | None = None) -> set[int]:
    """Ordinary immutable/current/reference recipe, independent of prune timing."""
    active = {row['binding']['generation'] for row in registry['service_references'].values()
        if row['status'] != 'released'}
    return {0, generation, *range(10, generation + 1, 10), *snapshot_pool, *active,
        *(row['generation'] for row in registry['evaluation_references'].values()), *(pending or ())}


def protected_files(records: dict[int, Json], registry: Json, *, generation: int,
                    operating: Json | None = None, pending: list[int] | None = None) -> dict[int, set[str]]:
    """Pure adapter of actual new registry references to the unchanged recipes."""
    active = {key: row['binding'] for key, row in registry['service_references'].items()
        if row['status'] != 'released'}
    if operating is not None:
        from gf_training.hpo_lineage import protected_hpo_files_for_operating
        return protected_hpo_files_for_operating(operating, [records[g] for g in range(generation + 1)],
            {'evaluation_references': registry['evaluation_references'], 'cache_references': active},
            [{'generation': g} for g in pending or ()])
    protected = {g: set(row['artifact_files']) for g, row in records.items()}
    predecessor = generation - 1
    references = {row['generation'] for row in active.values()} | {row['generation'] for row in registry['evaluation_references'].values()} | set(pending or ())
    if predecessor > 0 and predecessor % 10 and predecessor not in references:
        protected[predecessor] = {'inference.json'}
    return protected


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


class RustArtifactRetirement:
    """One actual owner's private same-filesystem move, never replayed from JSON."""
    def __init__(self, registry: Any, generation: int, name: str, row: Json, deadline: float) -> None:
        from gf_training.rust_player_registry import RustRunRegistry
        if type(registry) is not RustRunRegistry:
            raise ContractError('actual current Rust registry required')
        registry.require()
        if (registry.registry['generations'][str(generation)]['files'][name] is not row
                or registry.records[generation]['artifact_files'].get(name) != row['sha256']
                or row['status'] != 'retained'):
            raise ContractError('actual retained declared generation artifact required')
        self.registry, self.generation, self.name, self.row = registry, generation, name, row
        self.deadline = deadline
        self.retirement_id = uuid.uuid4().hex
        self.staging = '.prune-' + self.retirement_id
        self.fds: dict[str, OwnedDescriptor] = {}
        self._cleanup_children: list[Any] = []
        self.phase, self.closed = 'unstarted', False
        self.reservation: Any = None
        registry._retirement_owners.append(self)  # before fallible reservation/open

    def check(self) -> None:
        self.registry.require()
        from gf_training.group_pack_resources import remaining
        remaining(self.deadline)

    def _open(self, role: str, path: Any, flags: int, *, parent: str | None = None) -> int:
        fd = os.open(path, flags, **({'dir_fd': self.fds[parent].fd} if parent else {}))
        self.fds[role] = OwnedDescriptor(fd)
        return fd

    def _directory(self, role: str) -> list[int]:
        self.check()
        fd = self.fds[role].fd
        held = os.fstat(fd)
        if role == 'run':
            named = self.registry.directory.lstat()
        else:
            parent, name = {'parent': ('run', 'artifacts'), 'source': ('parent', f'{self.generation:06}'), 'stage': ('parent', self.staging)}[role]
            named = os.stat(name, dir_fd=self.fds[parent].fd, follow_symlinks=False)
        if not stat.S_ISDIR(held.st_mode) or _identity(held) != _identity(named):
            raise ContractError('retirement held directory entry replaced')
        return _identity(held)[:2]

    def _original(self, *, staged: bool = False) -> None:
        self.check()
        role = 'stage' if staged else 'source'
        held = os.fstat(self.fds['original'].fd)
        named = os.stat(self.name, dir_fd=self.fds[role].fd, follow_symlinks=False)
        expected = self.original_identity[:4] if staged else self.original_identity
        if (not stat.S_ISREG(held.st_mode) or _identity(held) != _identity(named)
                or _identity(held)[:len(expected)] != expected):
            raise ContractError('retirement held original replaced or changed')

    def _hash_original(self) -> str:
        checksum, offset = hashlib.sha256(), 0
        size = self.original_identity[2]
        while offset < size:
            self.check()
            block = os.pread(self.fds['original'].fd, min(65536, size - offset), offset)
            if not block:
                raise ContractError('retirement original shortened during hash')
            checksum.update(block)
            offset += len(block)
        if os.pread(self.fds['original'].fd, 1, size):
            raise ContractError('retirement original grew during hash')
        return checksum.hexdigest()

    def run(self) -> None:
        if self.phase != 'unstarted' or self.closed:
            raise ContractError('actual artifact retirement is single use; JSON/retry grants no authority')
        registry = self.registry
        try:
            self.check()
            self.reservation = registry.writer.parent_budget.reserve(128 * 1024)
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            self._open('run', registry.directory, flags)
            self._directory('run')
            self._open('parent', 'artifacts', flags, parent='run')
            parent = self._directory('parent')
            self._open('source', f'{self.generation:06}', flags, parent='parent')
            source = self._directory('source')
            self._open('original', self.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, parent='source')
            self.original_identity = _identity(os.fstat(self.fds['original'].fd))
            self._original()
            if self._hash_original() != self.row['sha256']:
                raise ContractError('retirement original SHA differs from declared artifact')
            self._original()
            os.mkdir(self.staging, 0o700, dir_fd=self.fds['parent'].fd)
            self.phase = 'held'
            self._open('stage', self.staging, flags, parent='parent')
            stage = self._directory('stage')
            if stage[0] != parent[0]:
                raise ContractError('retirement requires same filesystem staging')
            actual = registry.admission.observation
            common = {'schema_version': 1, 'run_id': registry.state['run_id'],
                'operation_id': actual['operation_id'], 'invocation_id': actual['invocation_id'],
                'retirement_id': self.retirement_id, 'generation': self.generation, 'file': self.name,
                'sha256': self.row['sha256'], 'original_identity': self.original_identity,
                'runtime_binding_sha256': registry.state['runtime_binding_sha256']}
            intent = {**common, 'kind': 'rust-player-artifact-retirement-intent-v1',
                'staging_basename': self.staging, 'parent_identity': parent,
                'source_directory_identity': source, 'stage_directory_identity': stage}
            validate_retirement_record(intent, receipt=False)
            directory = registry.directory / 'retirement' / 'artifacts' / self.retirement_id
            directory.mkdir(parents=True, exist_ok=False)
            path = directory / 'intent.json'
            checksum = registry.writer.write(path, intent, deadline=self.deadline, immutable=True)
            self.row.update(status='deleting', deletion_intent={'path': str(path.relative_to(registry.directory)), 'sha256': checksum})
            self.phase = 'intent'
            registry.writer.write(registry.directory / 'artifact-registry.json', registry.registry, deadline=self.deadline)
            for role in ('run', 'parent', 'source', 'stage'):
                self._directory(role)
            self._original()
            self.phase = 'rename_requested'
            os.rename(self.name, self.name, src_dir_fd=self.fds['source'].fd, dst_dir_fd=self.fds['stage'].fd)
            self._original(staged=True)
            self._directory('source')
            self._directory('stage')
            self.phase = 'unlink_requested'
            os.unlink(self.name, dir_fd=self.fds['stage'].fd)
            self.phase = 'unlinked'
            for role in ('source', 'stage'):
                self._directory(role)
                os.fsync(self.fds[role].fd)
            if os.listdir(self.fds['stage'].fd):
                raise ContractError('retirement held stage is not empty')
            self._directory('parent')
            os.fsync(self.fds['parent'].fd)
            self._directory('stage')
            os.rmdir(self.staging, dir_fd=self.fds['parent'].fd)
            self.phase = 'removed'
            self._directory('parent')
            os.fsync(self.fds['parent'].fd)
            self.close()  # positive closure is part of the receipt barrier
            receipt = {**common, 'kind': 'rust-player-artifact-retirement-receipt-v1',
                'intent_sha256': checksum, 'completed_unix': time.time()}
            validate_retirement_record(receipt, receipt=True)
            receipt_path = directory / 'receipt.json'
            receipt_sha = registry.writer.write(receipt_path, receipt, deadline=self.deadline, immutable=True)
            self.row.update(status='deleted', deletion_receipt={'path': str(receipt_path.relative_to(registry.directory)), 'sha256': receipt_sha})
            registry.writer.write(registry.directory / 'artifact-registry.json', registry.registry, deadline=self.deadline)
            self.phase = 'complete'
            registry._retirement_owners.remove(self)
        except BaseException as error:
            retain_cleanup(self, error)
            registry.failed = True
            if self.phase in ('unstarted', 'held'):
                try:
                    self.retire_unmoved()
                except BaseException as cleanup:  # noqa: BLE001 -- preserve original failure and nested FD custody.
                    retain_cleanup(self, cleanup)
                    error.add_note('retirement cleanup remains owned: ' + str(cleanup))
            raise

    def retire_unmoved(self) -> None:
        if self.phase not in ('unstarted', 'held', 'intent'):
            raise ContractError('destructive or uncertain retirement requires quarantine')
        if 'stage' in self.fds:
            # Do not call registry.require after marking the parent failed.
            from gf_training.structured_lineage import _directory_entry
            _directory_entry(self.fds['parent'].fd, self.staging, self.fds['stage'].fd)
            if os.listdir(self.fds['stage'].fd):
                raise ContractError('unmoved stage unexpectedly contains files')
            os.rmdir(self.staging, dir_fd=self.fds['parent'].fd)
            os.fsync(self.fds['parent'].fd)
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        close_resources(self, list(self.fds.values()))
        self.fds.clear()
        if self.reservation is not None:
            self.reservation.close()
        self.closed = True


def validate_retirement_pair(intent: Json, receipt: Json, *, intent_sha256: str) -> None:
    """Immutable DATA correspondence only; never deletion/recovery authority."""
    validate_retirement_record(intent, receipt=False)
    validate_retirement_record(receipt, receipt=True)
    if (any(intent[key] != receipt[key] for key in _COMMON - {'kind'})
            or receipt['intent_sha256'] != intent_sha256):
        raise ContractError('retirement receipt differs from same immutable intent')


def _retirement_history(control: Any, state: Json, prepared: Json, intent: Json,
                        pool: Any, loans: list[Any], deadline: float, *, hpo: bool) -> None:
    return _retirement_history_data(state, prepared, intent, pool, loans, deadline, hpo=hpo,
        parent_directory=Path(control.operation['parent_directory']), record_root=control.root,
        session_finish=control.launch['session_deadlines']['finish_bits'])


def _retirement_history_data(state: Json, prepared: Json, intent: Json,
                        pool: Any, loans: list[Any], deadline: float, *, hpo: bool, parent_directory: Path,
                             record_root: Path, session_finish: str) -> None:
    from gf_common.rust_player import (
        validate_exit,
        validate_local_admission,
        validate_operation,
        validate_prepared,
    )
    from gf_training.rust_player_completion import _read
    from gf_training.rust_player_lineage import validate_completion, validate_run
    from gf_training.rust_player_supervision import read_control_record
    invocation = intent['invocation_id']
    entry = state['invocations'].get(invocation)
    if entry is None:
        raise ContractError('retirement invocation missing from actual history')
    root = record_root
    operation, operation_sha = read_control_record(Path(entry['operation_path']), root)
    admission, admission_sha = read_control_record(Path(entry['admission_path']), root)
    validate_operation(operation, session_finish=session_finish)
    validate_local_admission(admission)
    if (operation_sha != entry['operation_sha256'] or admission_sha != entry['admission_sha256']
            or operation['operation_id'] != intent['operation_id'] or admission['operation_id'] != intent['operation_id']
            or admission['invocation_id'] != invocation or operation['run_id'] != state['run_id']
            or admission['run_id'] != state['run_id'] or operation['parent_directory'] != str(parent_directory)
            or operation['config_sha256'] != state['config_sha256'] or admission['config_sha256'] != state['config_sha256']
            or operation['runtime_binding_sha256'] != intent['runtime_binding_sha256']
            or admission['runtime_binding_sha256'] != intent['runtime_binding_sha256']
            or operation['execution_request'] != entry['execution_request']
            or admission['producer'] != state['identity']['player_runtime']['producer']):
        raise ContractError('retirement actual historical operation/admission binding differs')
    if invocation == prepared['invocation_id']:
        if prepared['operation_id'] != intent['operation_id'] or prepared['admission_sha256'] != admission_sha:
            raise ContractError('current retirement differs from actual frozen Prepared')
        return
    prior_state, prior_state_sha = _read(Path(operation['parent_directory']) / f'state-{invocation}.json',
        pool.budget, deadline, loans=loans)
    validate_run(prior_state, hpo=hpo)
    prior, prior_sha = read_control_record(root / f'prepared-{invocation}.json', root)
    exit_record, exit_sha = read_control_record(root / f'exit-{invocation}.json', root)
    completion, _ = read_control_record(root / f'completion-{invocation}.json', root)
    validate_completion(completion)
    validate_prepared(prior, admission)
    validate_exit(exit_record, admission)
    if (prior_state['run_id'] != state['run_id'] or prior_state['status'] not in ('paused', 'complete')
            or prior_state['invocations'].get(invocation) != entry or prior['state_sha256'] != prior_state_sha
            or prior['state'] != prior_state['status'] or prior['generation_sha256'] != prior_state['generation_sha256']
            or exit_record['prepared_sha256'] != prior_sha or not exit_record['cooperative']
            or exit_record['wait']['exit_code'] != 0 or not exit_record['descendants_reaped'] or exit_record['signals']
            or completion['kind'] != 'rust-player-training-completion-v1'
            or completion['state'] != prior_state['status'] or completion['operation_sha256'] != operation_sha
            or completion['prepared_sha256'] != prior_sha or completion['exit_sha256'] != exit_sha
            or completion['state_sha256'] != prior_state_sha
            or completion['generation_sha256'] != prior_state['generation_sha256']
            or any(completion[key] != admission[key] for key in ('supervision_id', 'invocation_id',
                'operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256'))
            or completion['source_manifest_sha256'] != prior['source_manifest_sha256']
            or completion['committed_update'] != prior_state['updates']
            or prior['committed_update'] != prior_state['updates'] or prior['generation'] != prior_state['updates']
            or completion['audit_path'] != str(root / f'cold-audit-{invocation}.json')
            or completion['eligible'] != (prior_state['status'] == 'complete' and (
                not hpo or prior_state['used_decisions'] == prior_state['effective_config']['hpo_spec']['decision_budget']))):
        raise ContractError('retirement historical frozen Prepared/Exit/completion chain differs')
    audit, _ = _read(Path(completion['audit_path']), pool.budget, deadline, completion['audit_sha256'], loans=loans)
    _validate_historical_audit(audit, state=prior_state, prepared=prior, completion=completion)



def _validate_historical_audit(audit: Json, *, state: Json, prepared: Json, completion: Json) -> None:
    """Validate the historical result under the recorded Source verification mode."""
    from gf_common.domain import digest
    from gf_common.rust_ppo_source import source_mode
    from gf_models.structured_sealed import basename
    from gf_training.rust_ppo_pack import validate_pack_verification
    mode = source_mode(state['effective_config'])
    version = 2 if mode is not None else 1
    fields = {'schema_version', 'kind', 'run_id', 'invocation_id', 'operation_sha256',
        'prepared_sha256', 'exit_sha256', 'state_sha256', 'generation_sha256', 'updates',
        'saved_pair', 'runtime_binding_sha256'}
    if mode is not None:
        fields.update({'ppo_source', 'empty_pack_verification'})
    closed(audit, fields)
    if (type(audit['schema_version']) is not int or audit['schema_version'] != version
            or audit['kind'] != f'rust-player-cold-audit-v{version}' or audit.get('ppo_source') != mode
            or any(audit[key] != completion[key] for key in ('run_id', 'invocation_id',
                'operation_sha256', 'prepared_sha256', 'exit_sha256', 'state_sha256',
                'generation_sha256', 'runtime_binding_sha256'))):
        raise ContractError('retirement historical audit1 identity/chain differs')
    saved = closed(audit['saved_pair'], {'generation', 'inference_sha256', 'resume_sha256',
        'artifact_files', 'optimizer_steps', 'full_content_validated'})
    integer(saved['generation'], 0)
    integer(saved['optimizer_steps'], 0)
    files = saved['artifact_files']
    if type(files) is not dict:
        raise ContractError('historical audit artifact map required')
    for name, sha in files.items():
        basename(name)
        hex_identity(sha)
    if (saved['generation'] != state['updates'] or saved['optimizer_steps'] != state['optimizer_steps']
            or saved['full_content_validated'] is not True
            or files.get('inference.json') != saved['inference_sha256']
            or files.get('resume.json') != saved['resume_sha256']
            or digest(files) != prepared['artifacts_sha256']):
        raise ContractError('retirement historical audit saved result differs')
    updates = audit['updates']
    if type(updates) is not list or len(updates) != state['updates']:
        raise ContractError('historical audit must contain only every committed update')
    for generation, row in enumerate(updates, 1):
        fields = {'generation', 'manifest_sha256', 'numeric_sha256', 'learner_decisions',
            'episode_digests', 'source_set_sha256'}
        if mode is not None:
            fields.update({'ppo_source', 'pack_verification'})
        closed(row, fields)
        if mode is not None:
            if row['ppo_source'] != mode:
                raise ContractError('historical audit Source recipe differs')
            validate_pack_verification(row['pack_verification'], run_id=state['run_id'])
        integer(row['generation'], 1)
        integer(row['learner_decisions'], 0)
        for key in ('manifest_sha256', 'numeric_sha256', 'source_set_sha256'):
            hex_identity(row[key])
        if row['generation'] != generation or type(row['episode_digests']) is not list:
            raise ContractError('historical audit committed update order differs')
        for checksum in row['episode_digests']:
            hex_identity(checksum)
    marker = state['invocations'][completion['invocation_id']].get('empty_collection')
    if mode is not None:
        expected_empty = marker['pack_verification'] if marker is not None else None
        if audit['empty_pack_verification'] != expected_empty:
            raise ContractError('historical empty Pack verification differs')
    last_source = marker['manifest_sha256'] if marker is not None else updates[-1]['manifest_sha256'] if updates else None
    if last_source != completion['source_manifest_sha256']:
        raise ContractError('historical audit source manifest differs from completion')



def _protected_bytes(required: dict[int, set[str]], shared: set[int]) -> int:
    """Measured known policy mapping; names already owned by registry are shared."""
    import sys

    from gf_training.structured_rollout import _owned_bytes
    size = sys.getsizeof(required)
    for generation, names in required.items():
        size += _owned_bytes(generation, shared) + sys.getsizeof(names)
        size += sum(_owned_bytes(name, shared) for name in names)
    return size

def _cold_protected_files(parent_directory: Path, state: Json, registry: Json, pool: Any,
                          loans: list[Any], deadline: float, *, hpo: bool,
                          operating: Json | None) -> dict[int, set[str]]:
    """Fund compact truthful policy DATA, using the existing Source account."""
    import sys

    from gf_training.rust_player_completion import _read
    from gf_training.rust_player_lineage import validate_record_shape
    from gf_training.structured_rollout import _owned_bytes
    if hpo and operating is None:
        raise ContractError('cold HPO protection requires loaded operating data')
    root = parent_directory
    if not hpo:
        # Ordinary cadence/reference requirements need only the current pool;
        # historical generations are individually audited below, without a
        # redundant policy-planning read of every old manifest.
        current, _ = _read(root / 'generations' / f"{state['updates']:06}.json", pool.budget, deadline, loans=loans)
        validate_record_shape(current, 'generation')
        protected = _ordinary_protected_generations(registry, state['updates'], current['parent_controls']['snapshot_pool'])
        required = {g: set(registry['generations'][str(g)]['files']) if g in protected else {'inference.json'}
            for g in range(state['updates'] + 1)}
        del current, protected
        shared: set[int] = set()
        _owned_bytes(registry, shared)
        loans[-1].resize(_protected_bytes(required, shared))
        return required
    seen: set[int] = set()
    _owned_bytes(registry, seen)  # registry's existing loan owns shared file maps/names
    records = {}
    prior_index_size = 0
    mark = len(loans)
    for generation in range(state['updates'] + 1):
        record, _ = _read(root / 'generations' / f'{generation:06}.json', pool.budget, deadline, loans=loans)
        validate_record_shape(record, 'generation', hpo=hpo)
        summary = {'artifact_files': registry['generations'][str(generation)]['files'],
            'parent_controls': {'snapshot_pool': record['parent_controls']['snapshot_pool']},
            'used_decisions': record['used_decisions'] if hpo else 0}
        records[generation] = summary
        del record
        size = _owned_bytes(summary, seen)
        if id(generation) not in seen:
            size += sys.getsizeof(generation)
            seen.add(id(generation))
        current_index_size = sys.getsizeof(records)
        loans[-1].resize(size + current_index_size - prior_index_size)
        prior_index_size = current_index_size
    required = protected_files(records, registry, generation=state['updates'], operating=operating)

    # Account actual simultaneously live policy graphs, without charging shared
    # registry names twice. No extra whole Source allowance is introduced.
    shared = set()
    _owned_bytes(registry, shared)
    actual = _owned_bytes(records, shared) + _protected_bytes(required, shared)
    funded = sum(loan.size for loan in loans[mark:])
    if actual > funded:
        loans[-1].resize(loans[-1].size + actual - funded)
    records.clear()
    del summary
    shared.clear()
    _owned_bytes(registry, shared)
    remaining = _protected_bytes(required, shared)
    for loan in loans[mark:]:
        size = min(loan.size, remaining)
        loan.resize(size)
        remaining -= size
    return required


def audit_artifact_files(control: Any, state: Json, prepared: Json, registry: Json,
                        pool: Any, loans: list[Any], deadline: float, *, hpo: bool,
                        operating: Json | None = None) -> None:
    return audit_artifact_files_data(state, prepared, registry, pool, loans, deadline, hpo=hpo,
        operating=operating, parent_directory=Path(control.operation['parent_directory']),
        record_root=control.root, session_finish=control.launch['session_deadlines']['finish_bits'])


def audit_artifact_files_data(state: Json, prepared: Json, registry: Json,
                        pool: Any, loans: list[Any], deadline: float, *, hpo: bool,
                        operating: Json | None = None, parent_directory: Path,
                             record_root: Path, session_finish: str) -> None:
    """Audit retained originals and exact retired history under the caller's owner."""
    if set(registry['generations']) != {str(g) for g in range(state['updates'] + 1)}:
        raise ContractError('artifact registry generation inventory differs')
    protection_mark = len(loans)
    required = _cold_protected_files(parent_directory, state, registry, pool, loans, deadline,
        hpo=hpo, operating=operating)
    for generation in range(state['updates'] + 1):
        mark = len(loans)
        _audit_artifact_generation(parent_directory, state, prepared, registry, pool, loans, deadline,
            generation=generation, required=required, hpo=hpo, record_root=record_root, session_finish=session_finish)
        # The helper's decoded graphs have reached last use before refunding
        # their SAME charges. Errors keep both frames and loans for the caller.
        for loan in loans[mark:]:
            loan.close()
        del loans[mark:]
    del required
    for loan in loans[protection_mark:]:
        loan.close()
    del loans[protection_mark:]


def _audit_artifact_generation(parent_directory: Path, state: Json, prepared: Json, registry: Json,
                               pool: Any, loans: list[Any], deadline: float, *,
                               generation: int, required: dict[int, set[str]], hpo: bool,
                               record_root: Path, session_finish: str) -> None:
    from gf_models.structured_sealed import basename
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_completion import _read
    from gf_training.rust_player_lineage import validate_record_shape
    root = parent_directory
    record, _ = _read(root / 'generations' / f'{generation:06}.json', pool.budget, deadline, loans=loans)
    validate_record_shape(record, 'generation', hpo=hpo)
    entry = closed(registry['generations'][str(generation)], {'inference_sha256', 'snapshot_identity', 'files'})
    if (entry['inference_sha256'] != record['inference_sha256'] or entry['snapshot_identity'] != record['snapshot_identity']
            or set(entry['files']) != set(record['artifact_files'])):
        raise ContractError('artifact registry differs from declared generation map')
    directory = root / 'artifacts' / f'{generation:06}'
    for name, checksum in record['artifact_files'].items():
        basename(name)
        row = closed(entry['files'][name], {'sha256', 'status', 'deletion_intent', 'deletion_receipt'})
        if row['sha256'] != checksum:
            raise ContractError('artifact registry SHA differs from generation')
        if row['status'] == 'retained':
            if row['deletion_intent'] is not None or row['deletion_receipt'] is not None:
                raise ContractError('retained artifact has retirement records')
            original = OriginalFile(directory / name, deadline)
            try:
                if original.sha256 != checksum:
                    raise ContractError('retained artifact checksum differs')
                original.recheck()
            finally:
                original.close()
            continue
        if row['status'] != 'deleted' or name in required[generation]:
            raise ContractError('uncertain or required artifact cannot be imported as deleted')
        intent_ref = closed(row['deletion_intent'], {'path', 'sha256'})
        receipt_ref = closed(row['deletion_receipt'], {'path', 'sha256'})
        for ref in (intent_ref, receipt_ref):
            hex_identity(ref['sha256'])
            path = Path(ref['path'])
            if path.is_absolute() or '..' in path.parts:
                raise ContractError('retirement reference outside parent')
        mark = len(loans)
        intent, intent_sha = _read(root / intent_ref['path'], pool.budget, deadline, intent_ref['sha256'], loans=loans)
        receipt, _ = _read(root / receipt_ref['path'], pool.budget, deadline, receipt_ref['sha256'], loans=loans)
        validate_retirement_pair(intent, receipt, intent_sha256=intent_sha)
        stem = Path('retirement') / 'artifacts' / intent['retirement_id']
        if (intent_ref['path'] != str(stem / 'intent.json') or receipt_ref['path'] != str(stem / 'receipt.json')
                or intent['run_id'] != state['run_id'] or intent['generation'] != generation or intent['file'] != name
                or intent['sha256'] != checksum or intent['runtime_binding_sha256'] != state['runtime_binding_sha256']):
            raise ContractError('retirement exact record path/generation/runtime differs')
        history_mark = len(loans)
        _retirement_history_data(state, prepared, intent, pool, loans, deadline, hpo=hpo,
            parent_directory=parent_directory, record_root=record_root, session_finish=session_finish)
        for loan in loans[history_mark:]:
            loan.close()
        del loans[history_mark:]
        # Consequence checks only; absence never authorizes mutation.
        _audit_deleted_absence(root, generation, name, intent)
        del intent, receipt
        for loan in loans[mark:]:
            loan.close()
        del loans[mark:]


class _RetirementDirectoryAudit:
    """Read-only held-directory consequence checks; no mutation authority."""
    def __init__(self) -> None:
        self._cleanup_children: list[Any] = []
        self.fds: list[OwnedDescriptor] = []

    def close(self) -> None:
        close_resources(self, self.fds)
        self.fds.clear()


def _audit_deleted_absence(root: Path, generation: int, name: str, intent: Json) -> None:
    owner = _RetirementDirectoryAudit()
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        # Check every physical ancestor, then keep exact artifacts/source FDs.
        from gf_training.group_pack_source import _no_symlink_path
        parent = _no_symlink_path(root / 'artifacts')
        parent_fd = os.open(parent, flags)
        owner.fds.append(OwnedDescriptor(parent_fd))
        source_fd = os.open(f'{generation:06}', flags, dir_fd=parent_fd)
        owner.fds.append(OwnedDescriptor(source_fd))
        from gf_training.structured_lineage import _directory_entry
        _directory_entry(parent_fd, f'{generation:06}', source_fd)
        held_parent = os.fstat(parent_fd)
        if (_identity(held_parent)[:2] != intent['parent_identity']
                or _identity(parent.lstat()) != _identity(held_parent)
                or _identity(os.fstat(source_fd))[:2] != intent['source_directory_identity']):
            raise ContractError('retirement original held directory identity differs')
        for fd, member in ((source_fd, name), (parent_fd, intent['staging_basename'])):
            try:
                os.stat(member, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ContractError('retirement artifact/staging pathname remains present')
        _directory_entry(parent_fd, f'{generation:06}', source_fd)
        if _identity(parent.lstat()) != _identity(os.fstat(parent_fd)):
            raise ContractError('retirement artifacts directory replaced during absence check')
        owner.close()
    except BaseException as error:
        retain_cleanup(owner, error)
        raise
