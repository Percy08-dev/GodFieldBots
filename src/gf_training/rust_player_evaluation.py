"""Rust-parent evaluation DATA and artifact references, without training authority."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gf_common.artifacts import ArtifactPhaseAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.rust_player import hex_identity, text
from gf_models.structured_sealed import SealedScan
from gf_training.group_pack_resources import StorageBudget, close_resources
from gf_training.group_pack_source import OriginalFile
from gf_training.rust_player_completion import (
    _capture_failure_graph,
    _clear_failure_frames,
    _discard_failure_graph,
)

_PARENT_FIELDS = {'schema_version', 'kind', 'origin', 'encoder', 'run_id', 'run_path',
    'training_seed', 'config_sha256', 'state_sha256', 'generation', 'generation_sha256',
    'generations_sha256', 'inference_sha256', 'used_decisions', 'effective_settings_sha256',
    'runtime_binding_sha256', 'completion_receipt_sha256', 'prepared_sha256', 'exit_sha256',
    'source_validation_sha256'}
_SEED_FIELDS = {'schema_version', 'kind', 'proposal_sha256', 'training_seed', 'run_path',
    'run_id', 'config_sha256', 'state_sha256', 'generation', 'generation_sha256',
    'inference_sha256', 'used_decisions', 'source_receipt_sha256', 'gpu_termination_sha256',
    'runtime_binding_sha256', 'completion_receipt_sha256', 'prepared_sha256', 'exit_sha256'}


def _receipt_fields(row: Json) -> None:
    for key in ('run_id', 'run_path'):
        text(row[key])
    path = Path(row['run_path'])
    if not path.is_absolute() or '..' in path.parts:
        raise ContractError('absolute original Rust parent path required')
    for key in ('training_seed', 'generation', 'used_decisions'):
        integer(row[key], 0)
    for key, value in row.items():
        if key.endswith('_sha256'):
            hex_identity(value)


def validate_parent_metadata(value: Json) -> Json:
    """Existing metadata3 selector/content only; grants no Parent or Source."""
    row = closed(value, _PARENT_FIELDS)
    if (type(row['schema_version']) is not int or row['schema_version'] != 3
            or row['kind'] != 'hpo-rust-parent-metadata-validation'
            or row['origin'] != 'producer-owned-v1' or row['encoder'] not in ('canonical-python', 'canonical-rust')):
        raise ContractError('explicit Rust parent metadata3 required')
    _receipt_fields(row)
    return row


def validate_seed_training(value: Json) -> Json:
    """Closed seed2 DATA; shape-equivalent JSON cannot issue live completion."""
    row = closed(value, _SEED_FIELDS)
    if (type(row['schema_version']) is not int or row['schema_version'] != 2
            or row['kind'] != 'hpo-rust-seed-training'):
        raise ContractError('explicit Rust closed seed-training2 required')
    _receipt_fields(row)
    return row


def validate_seed_binding(value: Json, metadata: Json, *, source_sha256: str,
                          termination_sha256: str) -> None:
    """Cross-record DATA correspondence only; qualification needs actual owners."""
    validate_seed_training(value)
    validate_parent_metadata(metadata)
    hex_identity(source_sha256)
    hex_identity(termination_sha256)
    keys = ('training_seed', 'run_path', 'run_id', 'config_sha256', 'state_sha256', 'generation',
        'generation_sha256', 'inference_sha256', 'used_decisions', 'runtime_binding_sha256',
        'completion_receipt_sha256', 'prepared_sha256', 'exit_sha256')
    if (any(value[key] != metadata[key] for key in keys)
            or value['source_receipt_sha256'] != source_sha256
            or value['gpu_termination_sha256'] != termination_sha256):
        raise ContractError('closed Rust seed differs from exact parent/source/physical records')


def validate_parent_binding(metadata: Json, state: Json, *, run_path: str) -> None:
    """Scalar DATA match; does not replace full Run/Source/saved-state validation."""
    validate_parent_metadata(metadata)
    spec = state['effective_config']['hpo_spec']
    if (state['run_variant'] != 'hpo-ppo-owned-rust-player-v1' or state['status'] != 'complete'
            or spec['schema_version'] != 6 or state['used_decisions'] != spec['decision_budget']
            or metadata['encoder'] != spec['encoder']
            or metadata['run_path'] != run_path or metadata['run_id'] != state['run_id']
            or metadata['training_seed'] != state['seed'] or metadata['generation'] != state['updates']
            or metadata['used_decisions'] != state['used_decisions']
            or metadata['config_sha256'] != state['config_sha256']
            or metadata['generation_sha256'] != state['generation_sha256']
            or metadata['runtime_binding_sha256'] != state['runtime_binding_sha256']
            or metadata['effective_settings_sha256'] != spec['effective_settings_sha256']):
        raise ContractError('complete exact-D Rust parent metadata correspondence required')


def validate_reference_binding(value: Json, state: Json, generation: Json, *,
                               generation_sha256: str) -> Json:
    """Existing artifact-only reference1 identity, without old scan authority."""
    import re

    row = closed(value, {'run_id', 'token', 'generation', 'inference_sha256', 'generation_sha256'})
    integer(row['generation'], 0)
    for key in ('inference_sha256', 'generation_sha256'):
        hex_identity(row[key])
    if (type(row['token']) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', row['token']) is None
            or row['run_id'] != state['run_id'] or row['generation'] != generation['generation']
            or row['inference_sha256'] != generation['inference_sha256']
            or row['generation_sha256'] != generation_sha256):
        raise ContractError('Rust evaluation reference1 immutable binding differs')
    return row


class _RetainedInferenceFiles:
    """Independent readonly originals; owns no weight heap or Source authority."""
    def __init__(self, directory: Path, expected: Json, *, budget: StorageBudget,
                 parent_budget: ArtifactPhaseAdmission, deadline: float | None = None) -> None:
        if (type(budget) is not StorageBudget or type(parent_budget) is not ArtifactPhaseAdmission
                or parent_budget.source is not budget):
            raise ContractError('actual SAME evaluation account/parent phase required')
        self.parent_budget, self.budget = parent_budget, budget
        self.deadline = budget.deadline if deadline is None else min(deadline, budget.deadline)
        self._held: dict[str, OriginalFile] = {}
        self._pending_original: OriginalFile | None = None
        self._cleanup_children: list[Any] = []
        self._metadata_charge = self._hash_charge = None
        self._closed, self._failed = False, False
        self._files = b''
        try:
            from gf_models.structured_sealed import basename
            if type(expected) is not dict or 'inference.json' not in expected:
                raise ContractError('declared inference original map required')
            for name, sha in expected.items():
                basename(name)
                hex_identity(sha)
                if name != 'inference.json' and not name.startswith('weights-'):
                    raise ContractError('evaluation owns inference files only')
            self._metadata_charge = parent_budget.reserve(32 * json_bound(expected) + 65536)
            self._files = canonical(expected).encode()
            # OriginalFile.hash has two simultaneously live64KiB read buffers;
            # failed constructor tracebacks keep this charge until explicit close.
            self._hash_charge = parent_budget.reserve(128 * 1024)
            for name, sha in expected.items():
                self._pending_original = OriginalFile(directory / name, self.deadline)
                original = self._pending_original
                self._held[name] = original
                self._pending_original = None
                if original.sha256 != sha:
                    raise ContractError('evaluation original checksum differs from declared inference')
                original.recheck()
            self._hash_charge.close()
            self._hash_charge = None
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def require(self) -> None:
        if self._closed or self._failed:
            raise ContractError('live independent readonly inference DATA required')
        try:
            self._hash_charge = self.parent_budget.reserve(128 * 1024)
            for original in self._held.values():
                original.recheck()
            self._hash_charge.close()
            self._hash_charge = None
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def _handoff_deadline(self) -> None:
        # These receiving originals belong to the already-declared evaluation
        # account. Only the live Parent factory calls this after bounded handoff.
        self.deadline = self.budget.deadline
        for original in self._held.values():
            original.deadline = self.deadline

    @property
    def files(self) -> Json:
        self.require()
        return json.loads(self._files)

    def close(self) -> None:
        if self._closed:
            return
        _clear_failure_frames(self)
        close_resources(self, [*self._held.values(), self._pending_original])
        self._held.clear()
        self._pending_original = None
        self._files = b''
        if self._hash_charge is not None:
            self._hash_charge.close()
            self._hash_charge = None
        if self._metadata_charge is not None:
            self._metadata_charge.close()
            self._metadata_charge = None
        self._closed = True
        _discard_failure_graph(self)

    def __reduce__(self) -> Any:
        raise TypeError('independent inference original DATA ownership cannot be copied')


_COMPLETED_AUDIT_ISSUER = object()


class RustCompletedParentAudit:
    """Private current-owned DATA audit arm; never a recreated cold Parent.

    The authenticated C1 bootstrap factory is the sole future caller of this
    constructor. There is deliberately no public issuance factory until that
    concrete bootstrap/campaign-custody API is available.
    """
    def __init__(self, issuer: object, control: Any, ownership: Any, *, loaded: Any,
                 pool: Any, parent_budget: ArtifactPhaseAdmission, deadline: float,
                 record_root: Path) -> None:
        import math

        from gf_training.group_pack_pool import PackedArtifactPool
        from gf_training.lineage import require_hpo_run_owner
        from gf_training.rust_player_supervision import HostControl
        if (issuer is not _COMPLETED_AUDIT_ISSUER or type(control) is not HostControl
                or type(pool) is not PackedArtifactPool or type(parent_budget) is not ArtifactPhaseAdmission
                or parent_budget.source is not pool.budget):
            raise ContractError('actual current campaign/RunOwnership/evaluation pool DATA issuer required')
        control._require()
        self.ownership = require_hpo_run_owner(ownership.directory, ownership)
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or deadline > pool.deadline or deadline > control._finish()
                or not isinstance(record_root, Path)
                or not record_root.is_absolute() or record_root.parent != control.root.parent):
            raise ContractError('current evaluation deadline and bound historical record root required')
        hex_identity(record_root.name, 32)
        self.control, self.loaded = control, loaded
        self.pool, self.budget, self.parent_budget = pool, pool.budget, parent_budget
        self.deadline, self.record_root = deadline, record_root
        self._phase, self._closed, self._failed = 'BOUND', False, False
        self._pool_hold: Any = None
        self._pending_child: Any = None
        self._cleanup_children: list[Any] = []
        self._loans: list[Any] = []
        self._held: list[OriginalFile] = []
        self._pending_original: OriginalFile | None = None
        self._pending_loan: Any = None
        self._children: list[Any] = []
        self.state: Json = {}
        self.current: Json = {}
        self.metadata: Json = {}
        self.campaign_id = ''
        self._headers: dict[Path, Json] = {}
        self._header_shas: dict[Path, str] = {}
        self._header_charge: Any = None
        self._scan: _FundedEvaluationScan | None = None
        self._artifact: Any = None
        self.originals: _RetainedInferenceFiles | None = None
        try:
            self._pool_hold = self.pool.hold_close(self)
            self.require()
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def require(self) -> None:
        from gf_training.group_pack_resources import remaining
        from gf_training.lineage import require_hpo_run_owner
        self.control._require()
        require_hpo_run_owner(self.ownership.directory, self.ownership)
        self.pool.check()
        remaining(self.deadline)
        if self._closed or self._failed or self._phase not in ('BOUND', 'AUDITING', 'READY'):
            raise ContractError('actual current-owned completed-parent DATA audit required')
        for original in self._held:
            original.check_identity()

    def read_data(self, path: Path, expected_sha256: str | None = None, *, protocol: bool = True) -> Json:
        self.require()
        try:
            if path in self._headers:
                if expected_sha256 is not None and self._header_shas[path] != expected_sha256:
                    raise ContractError('completed audit original digest differs')
                return self._headers[path]
            if self._header_charge is None:
                self._header_charge = self.parent_budget.reserve(128 * 1024)
            self._pending_original = OriginalFile(path, self.deadline)
            original = self._pending_original
            self._held.append(original)
            self._pending_original = None
            if expected_sha256 is not None and original.sha256 != expected_sha256:
                raise ContractError('completed audit original digest differs')
            # Exact identical immutable bytes share the decoded graph/charge.
            for other, sha in self._header_shas.items():
                if sha == original.sha256:
                    self._headers[path], self._header_shas[path] = self._headers[other], sha
                    return self._headers[path]
            self._pending_loan = self.budget.reserve(32 * original.identity.st_size + 65536)
            self._loans.append(self._pending_loan)
            self._pending_loan = None
            raw = original.read(self.parent_budget.limit)
            value = json.loads(raw)
            encoded = canonical(value).encode() + (b'' if protocol else b'\n')
            if type(value) is not dict or raw != encoded:
                raise ContractError('canonical completed-audit original data required')
            self._headers[path], self._header_shas[path] = value, original.sha256
            return value
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def retain_evaluation_metadata(self, manifest: Json, proof: Json) -> Any:
        self.require()
        try:
            self._pending_child = _RetainedEvaluationMetadata(manifest, proof, pool=self.pool, parent_budget=self.parent_budget)
            self._children.append(self._pending_child)
            result = self._pending_child
            self._pending_child = None
            return result
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def audit_payload(self, state: Json, metadata: Json, prepared: Json, *, session_finish: str) -> None:
        from gf_training.rust_player_completion import (
            _audit_committed_data,
            _audit_retirement_data,
            _audit_saved_data,
        )
        from gf_training.rust_player_lineage import validate_run
        self.require()
        if self._phase != 'BOUND':
            raise ContractError('completed DATA audit is single use')
        validate_run(state, hpo=True)
        validate_parent_binding(metadata, state, run_path=str(self.ownership.directory))
        if (state['config_sha256'] != self.loaded.sha256
                or canonical(state['effective_config']) != canonical(self.loaded.effective)):
            raise ContractError('completed audit LoadedConfig differs from actual parent')
        self.state, self.metadata, self._phase = state, metadata, 'AUDITING'
        try:
            _audit_retirement_data(state, prepared, self.pool, self._loans, self.deadline, hpo=True,
                operating=self.loaded.config.hpo_spec['operating'], parent_directory=self.ownership.directory,
                record_root=self.record_root, session_finish=session_finish)
            current, generation_sha, generations_sha, updates, _empty_pack = _audit_committed_data(self, self.ownership, self.pool,
                self.loaded, state, prepared, self.deadline, control_loans=self._loans)
            if generation_sha != metadata['generation_sha256'] or generations_sha != metadata['generations_sha256']:
                raise ContractError('full independent completed generation chain differs')
            self.current = current
            limit = min(self.parent_budget.limit - self.parent_budget.used, self.budget.limit - self.budget.used)
            self._scan = _FundedEvaluationScan(self.pool, self.parent_budget, max_owned_bytes=limit, deadline=self.deadline)
            self._children.append(self._scan)
            artifact = _audit_saved_data(self.ownership.directory, self.loaded, state, current, self._scan)
            self._artifact = artifact
            self._scan.finalize()
            self._scan.settle()
            audit = self.read_data(self.record_root / f"cold-audit-{prepared['invocation_id']}.json", metadata['source_validation_sha256'])
            if (canonical(audit['updates']) != canonical(updates)
                    or audit['saved_pair']['artifact_files'] != artifact.files
                    or audit['saved_pair']['optimizer_steps'] != state['optimizer_steps']
                    or artifact.files['inference.json'] != metadata['inference_sha256']):
                raise ContractError('independent Raw/numeric/full saved content differs from closed audit')
            expected = {'inference.json': artifact.files['inference.json'], **artifact.inference['weights']['files']}
            self.originals = _RetainedInferenceFiles(self.ownership.directory / 'artifacts' / f"{state['updates']:06}",
                expected, budget=self.budget, parent_budget=self.parent_budget, deadline=self.deadline)
            self._children.append(self.originals)
            for original in self._held:
                original.recheck()
            validate_reference_inventory(self.ownership, state_sha256=metadata['state_sha256'],
                pool=self.pool, parent_budget=self.parent_budget, deadline=self.deadline)
            self.require()
            self._phase = 'READY'
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def close(self) -> None:
        if self._closed:
            return
        _clear_failure_frames(self)
        self._artifact = None
        close_resources(self, [*self._children, self._pending_child, self.originals, self._scan,
            *self._held, self._pending_original])
        self._pending_child = None
        self._pending_original = None
        self._children.clear()
        self._held.clear()
        self.state.clear()
        self.current.clear()
        self.metadata.clear()
        self._headers.clear()
        self._header_shas.clear()
        if self._pending_loan is not None:
            self._pending_loan.close()
            self._pending_loan = None
        for loan in self._loans:
            loan.close()
        self._loans.clear()
        if self._header_charge is not None:
            self._header_charge.close()
            self._header_charge = None
        for hold in tuple(self.pool._close_holds):
            if hold.owner is self:
                hold.retire(self)
        self._closed = True
        _discard_failure_graph(self)

    def __reduce__(self) -> Any:
        raise TypeError('actual completed-parent DATA audit ownership cannot be serialized')


class _FundedEvaluationScan(SealedScan):
    """Existing scan estimates backed by the SAME actual evaluation account."""
    def __init__(self, pool: Any, parent_budget: ArtifactPhaseAdmission, *,
                 max_owned_bytes: int, deadline: float) -> None:
        from gf_training.group_pack_pool import PackedArtifactPool
        if (type(pool) is not PackedArtifactPool or type(parent_budget) is not ArtifactPhaseAdmission
                or parent_budget.source is not pool.budget or deadline > pool.deadline):
            raise ContractError('actual SAME evaluation pool/phase and original deadline required')
        self.pool, self.parent_budget = pool, parent_budget
        self._coupons: list[Any] = []
        self._pending_coupon: Any = None
        self._cleanup_children: list[Any] = []
        self._funded = 0
        self._closed, self._failed = False, False
        super().__init__(max_owned_bytes=max_owned_bytes, deadline=deadline)

    def check(self) -> None:
        if self._closed or self._failed:
            raise ContractError('actual healthy funded evaluation scan required')
        self.pool.check()
        super().check()

    def charge(self, size: int, *, temporary: bool = False) -> None:
        before = self.retained
        try:
            super().charge(size, temporary=temporary)
            required = before + size
            if required > self._funded:
                # Preserve the actual coupon before the list transfer can fail.
                self._pending_coupon = self.parent_budget.reserve(required - self._funded)
                self._coupons.append(self._pending_coupon)
                self._funded = required
                self._pending_coupon = None
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def settle(self) -> None:
        """Caller has positively retired the construction/final-check scratch."""
        self.check()
        while self._funded > self.retained:
            coupon = self._coupons[-1]
            excess = self._funded - self.retained
            if coupon.size <= excess:
                size = coupon.size
                coupon.close()
                self._coupons.pop()
                self._funded -= size
            else:
                coupon.shrink(coupon.size - excess)
                self._funded -= excess

    def __reduce__(self) -> Any:
        raise TypeError('actual funded scan custody cannot be copied or serialized')

    def close(self) -> None:
        if self._closed:
            return
        self._failed = True
        _clear_failure_frames(self)
        close_resources(self, [])
        self.files.clear()
        self.inventories.clear()
        self.directory_inventories.clear()
        if self._pending_coupon is not None:
            self._pending_coupon.close()
            self._pending_coupon = None
        while self._coupons:
            coupon = self._coupons[-1]
            size = coupon.size
            coupon.close()
            self._coupons.pop()
            self._funded -= size
        self.retained = 0
        self._closed = True
        _discard_failure_graph(self)


class _ReferenceJournal:
    """Actual RunOwnership controls Rust reference1 I/O; never Source authority."""
    def __init__(self, ownership: Any, *, pool: Any, parent_budget: ArtifactPhaseAdmission,
                 deadline: float) -> None:
        from gf_training.group_pack_pool import PackedArtifactPool
        from gf_training.hpo_owned_lineage import OwnedControlWriter
        from gf_training.lineage import RunOwnership, require_hpo_run_owner
        if type(ownership) is not RunOwnership:
            raise ContractError('actual current RunOwnership required')
        self.ownership = require_hpo_run_owner(ownership.directory, ownership)
        if (type(pool) is not PackedArtifactPool or type(parent_budget) is not ArtifactPhaseAdmission
                or parent_budget.source is not pool.budget or deadline > pool.deadline):
            raise ContractError('actual SAME reference pool/phase/deadline required')
        self.pool, self.parent_budget, self.deadline = pool, parent_budget, deadline
        self.root = ownership.directory
        self._held: dict[Path, OriginalFile] = {}
        self._pending_original: OriginalFile | None = None
        self._pending_loan: Any = None
        self._objects: dict[Path, Json] = {}
        self._shas: dict[Path, str] = {}
        self._loans: list[Any] = []
        self._cleanup_children: list[Any] = []
        self._closed = False
        self.writer = OwnedControlWriter(ownership, budget=pool.budget, limit=parent_budget.limit)
        # This fresh writer has not reserved or written anything. Its actual
        # writes join the caller's SAME already-admitted metadata phase.
        self.writer.parent_budget = parent_budget
        try:
            pool.hold_close(self)
            self.reserve(128 * 1024)
        except BaseException as error:
            _capture_failure_graph(self, error)
            raise

    def require(self) -> None:
        from gf_training.group_pack_resources import remaining
        from gf_training.lineage import require_hpo_run_owner
        require_hpo_run_owner(self.root, self.ownership)
        self.pool.check()
        remaining(self.deadline)
        if self._closed:
            raise ContractError('actual reference I/O owner is closed')

    def reserve(self, size: int) -> None:
        self._pending_loan = self.parent_budget.reserve(size)
        self._loans.append(self._pending_loan)
        self._pending_loan = None

    def read(self, path: Path, expected: str | None = None) -> Json:
        self.require()
        try:
            if path not in self._objects:
                self._pending_original = OriginalFile(path, self.deadline)
                original = self._pending_original
                self._held[path] = original
                self._pending_original = None
                self.reserve(32 * original.identity.st_size + 65536)
                raw = original.read(self.parent_budget.limit)
                value = json.loads(raw)
                if type(value) is not dict or raw != (canonical(value) + '\n').encode():
                    raise ContractError('canonical completed Rust reference/parent data required')
                self._objects[path], self._shas[path] = value, original.sha256
            self._held[path].recheck()
            if expected is not None and self._shas[path] != expected:
                raise ContractError('Rust reference immutable original digest differs')
            return self._objects[path]
        except BaseException as error:
            _capture_failure_graph(self, error)
            raise

    def __reduce__(self) -> Any:
        raise TypeError('actual reference I/O custody cannot be copied or serialized')

    def close(self) -> None:
        if self._closed:
            return
        _clear_failure_frames(self)
        close_resources(self, [*self._held.values(), self._pending_original])
        self._pending_original = None
        if self.writer._uncertain_closes:
            raise ContractError('actual reference writer descriptor retirement remains uncertain')
        self._held.clear()
        self._objects.clear()
        self._shas.clear()
        self.writer._completed.clear()
        self.writer.files.clear()
        if self._pending_loan is not None:
            self._pending_loan.close()
            self._pending_loan = None
        for loan in self._loans:
            loan.close()
        self._loans.clear()
        for hold in tuple(self.pool._close_holds):
            if hold.owner is self:
                hold.retire(self)
        self._closed = True
        _discard_failure_graph(self)


def _reference_journal(reader: _ReferenceJournal, state: Json, registry: Json) -> None:
    import math
    import os
    import re
    import stat
    import uuid

    from gf_training.hpo_lineage import REFERENCE_DIRECTORY, REFERENCE_FILES
    from gf_training.rust_player_lineage import validate_record_shape
    root = reader.root / REFERENCE_DIRECTORY
    names = []
    if os.path.lexists(root):
        before = root.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise ContractError('Rust reference namespace must be a nonsymlink directory')
        with os.scandir(root) as entries:
            for entry in entries:
                reader.reserve(1024 + 4 * len(entry.name))
                if not entry.is_dir(follow_symlinks=False) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', entry.name) is None:
                    raise ContractError('foreign Rust reference namespace')
                names.append(entry.name)
        for token in sorted(names):
            folder = root / token
            files = set()
            with os.scandir(folder) as entries:
                for entry in entries:
                    reader.reserve(512)
                    if entry.name not in REFERENCE_FILES or not entry.is_file(follow_symlinks=False):
                        raise ContractError('foreign Rust reference journal file')
                    files.add(entry.name)
            if ('acquire.intent.json' not in files or 'acquire.receipt.json' not in files
                    or ('release.intent.json' in files) != ('release.receipt.json' in files)):
                raise ContractError('interrupted Rust reference cannot replay JSON intent')
            binding = None
            for phase in ('acquire', 'release'):
                intent_path = folder / (phase + '.intent.json')
                if intent_path.name not in files:
                    continue
                intent = closed(reader.read(intent_path), {'schema_version', 'kind', 'operation_id', 'binding'}
                    | ({'acquire_receipt_sha256'} if phase == 'release' else set()))
                integer(intent['schema_version'], 1, 1)
                if intent['kind'] != 'hpo-evaluation-reference-' + phase + '-intent':
                    raise ContractError('Rust evaluation reference intent kind differs')
                try:
                    if str(uuid.UUID(intent['operation_id'])) != intent['operation_id']:
                        raise ValueError('noncanonical UUID')
                except (ValueError, TypeError, AttributeError) as error:
                    raise ContractError('reference operation UUID required') from error
                item = intent['binding']
                integer(item['generation'], 0, state['updates'])
                generation = reader.read(reader.root / 'generations' / f"{item['generation']:06}.json", item['generation_sha256'])
                validate_record_shape(generation, 'generation', hpo=True)
                validate_reference_binding(item, state, generation, generation_sha256=item['generation_sha256'])
                if item['token'] != token or binding is not None and item != binding:
                    raise ContractError('Rust reference journal binding changed')
                binding = item
                if phase == 'release' and intent['acquire_receipt_sha256'] != reader._shas[folder / 'acquire.receipt.json']:
                    raise ContractError('reference release changed original acquisition')
                receipt = closed(reader.read(folder / (phase + '.receipt.json')), {'schema_version', 'kind',
                    'binding', 'intent_sha256', 'registry_sha256', 'confirmed_unix'})
                integer(receipt['schema_version'], 1, 1)
                if (receipt['kind'] != 'hpo-evaluation-reference-' + phase + '-receipt'
                        or receipt['binding'] != binding or receipt['intent_sha256'] != reader._shas[intent_path]):
                    raise ContractError('reference receipt differs from actual immutable intent')
                hex_identity(receipt['registry_sha256'])
                when = receipt['confirmed_unix']
                if type(when) not in (int, float) or not math.isfinite(when) or when <= 0:
                    raise ContractError('actual reference completion time required')
            assert binding is not None
            current = registry['evaluation_references'].get(token)
            if ('release.receipt.json' in files and current is not None
                    or 'release.receipt.json' not in files and current != {
                        'generation': binding['generation'], 'inference_sha256': binding['inference_sha256']}):
                raise ContractError('Rust reference ledger differs from completed journal')
        after = root.lstat()
        if (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns) != (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns):
            raise ContractError('Rust reference namespace changed during read')
    if set(registry['evaluation_references']) - set(names):
        raise ContractError('Rust reference ledger lacks original intent/receipt')



def validate_reference_inventory(ownership: Any, *, state_sha256: str, pool: Any,
                                 parent_budget: ArtifactPhaseAdmission, deadline: float) -> None:
    """Read-only completed reference1 validation under actual current ownership."""
    from gf_training.rust_player_lineage import validate_record_shape, validate_run
    reader = _ReferenceJournal(ownership, pool=pool, parent_budget=parent_budget, deadline=deadline)
    try:
        state = validate_run(reader.read(reader.root / 'run.json', state_sha256), hpo=True)
        registry = validate_record_shape(reader.read(reader.root / 'artifact-registry.json'), 'registry', hpo=True)
        if registry['run_id'] != state['run_id'] or registry['runtime_binding_sha256'] != state['runtime_binding_sha256']:
            raise ContractError('Rust reference inventory run/runtime differs')
        _reference_journal(reader, state, registry)
        del state, registry
    except BaseException as error:
        _capture_failure_graph(reader, error)
        raise
    reader.close()

def _reference_transition(ownership: Any, binding: Json, phase: str, *, state_sha256: str,
                          pool: Any, parent_budget: ArtifactPhaseAdmission, deadline: float,
                          acquisition_sha256: str | None = None) -> Any:
    """Artifact-only actual-owner transaction; never replay incomplete JSON intent."""
    import os
    import time
    import uuid

    from gf_training.hpo_lineage import REFERENCE_DIRECTORY
    from gf_training.hpo_retention import HpoReferenceReceipt, _mkdir
    from gf_training.rust_player_lineage import validate_record_shape, validate_run
    if phase not in ('acquire', 'release'):
        raise ContractError('closed Rust evaluation reference phase required')
    reader = _ReferenceJournal(ownership, pool=pool, parent_budget=parent_budget, deadline=deadline)
    try:
        state = validate_run(reader.read(reader.root / 'run.json', state_sha256), hpo=True)
        generation = reader.read(reader.root / 'generations' / f"{binding['generation']:06}.json", binding['generation_sha256'])
        validate_record_shape(generation, 'generation', hpo=True)
        validate_reference_binding(binding, state, generation, generation_sha256=binding['generation_sha256'])
        registry_path = reader.root / 'artifact-registry.json'
        registry = validate_record_shape(reader.read(registry_path), 'registry', hpo=True)
        if (registry['run_id'] != state['run_id'] or registry['runtime_binding_sha256'] != state['runtime_binding_sha256']
                or generation['runtime_binding_sha256'] != state['runtime_binding_sha256']
                or generation['run_id'] != state['run_id']):
            raise ContractError('Rust reference parent/runtime/registry differs')
        _reference_journal(reader, state, registry)
        root = reader.root / REFERENCE_DIRECTORY
        folder = root / binding['token']
        if phase == 'acquire':
            if os.path.lexists(folder):
                raise ContractError('existing/interrupted Rust reference token cannot replay or change owner')
            row = registry['generations'][str(binding['generation'])]['files']
            expected = {'inference.json', *(name for name in generation['artifact_files'] if name.startswith('weights-'))}
            if any(row[name]['status'] != 'retained' or row[name]['sha256'] != generation['artifact_files'][name] for name in expected):
                raise ContractError('Rust reference requires retained inference and every weight shard')
            _mkdir(root, ownership=ownership)
            _mkdir(folder, ownership=ownership)
        else:
            acquisition = reader.read(folder / 'acquire.receipt.json', acquisition_sha256)
            if acquisition_sha256 is None or acquisition['binding'] != binding or os.path.lexists(folder / 'release.intent.json'):
                raise ContractError('actual original acquisition required; release token is terminal')
        reader.require()
        intent = {'schema_version': 1, 'kind': 'hpo-evaluation-reference-' + phase + '-intent',
            'operation_id': str(uuid.uuid4()), 'binding': dict(binding)}
        if phase == 'release':
            intent['acquire_receipt_sha256'] = acquisition_sha256
        intent_path = folder / (phase + '.intent.json')
        intent_sha = reader.writer.write(intent_path, intent, deadline=deadline, immutable=True)
        original = reader._held[registry_path]
        original.recheck()
        reader.writer.files[registry_path] = tuple(getattr(original.identity, key) for key in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
        original.close()
        del reader._held[registry_path]
        if phase == 'acquire':
            registry['evaluation_references'][binding['token']] = {
                'generation': binding['generation'], 'inference_sha256': binding['inference_sha256']}
        else:
            del registry['evaluation_references'][binding['token']]
        registry_sha = reader.writer.write(registry_path, registry, deadline=deadline)
        receipt = {'schema_version': 1, 'kind': 'hpo-evaluation-reference-' + phase + '-receipt',
            'binding': dict(binding), 'intent_sha256': intent_sha, 'registry_sha256': registry_sha,
            'confirmed_unix': time.time()}
        receipt_path = folder / (phase + '.receipt.json')
        checksum = reader.writer.write(receipt_path, receipt, deadline=deadline, immutable=True)
        reader.writer._completed[receipt_path].require(reader.writer, receipt_path, checksum)
        result = HpoReferenceReceipt(dict(binding), checksum, phase)
        del state, generation, registry, intent, receipt
        if phase == 'release':
            del acquisition
        else:
            del row, expected
    except BaseException as error:
        _capture_failure_graph(reader, error)
        raise
    reader.close()
    return result


def final_policy_binding(state: Json, metadata: Json) -> tuple[str, Any]:
    """Pure Model4 final binding from validated new-record DATA, not old authority."""
    from dataclasses import asdict

    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import RuntimePolicyBinding, digest
    from gf_common.structured_layout import layout_for_contract
    from gf_models.structured_binding import (
        configured_model_v4,
        input_binding_game_v3,
        policy_label_game_v4,
        safe_options_game_v4,
    )
    from gf_models.structured_config import InferenceBindingV2
    validate_parent_binding(metadata, state, run_path=metadata['run_path'])
    actions, rules = action_config_v3(state['effective_config']['actions']), card_instance_rules()
    configured = configured_model_v4(state['effective_config']['model'], layout=layout_for_contract(actions.action_contract))
    model = configured.architecture(layout=layout_for_contract(actions.action_contract))
    execution = state['execution']['inference']
    inference = InferenceBindingV2(**execution['binding'])
    path = Path(metadata['run_path']) / 'artifacts' / f"{metadata['generation']:06}"
    options = safe_options_game_v4({'version': '4', 'checkpoint': str(path),
        'expected_manifest_digest': metadata['inference_sha256'], 'model': asdict(model),
        'inference': asdict(inference), 'cuda_visible_devices': execution['cuda_visible_devices'],
        'action_config_hash': digest(asdict(actions)), **input_binding_game_v3(actions.action_contract, rules)},
        actions=actions, rules=rules)
    label = policy_label_game_v4(model, inference, metadata['inference_sha256'],
        action_contract=actions.action_contract, rules=rules, action_config_hash=digest(asdict(actions)),
        cuda_visible_devices=execution['cuda_visible_devices'])
    return label, RuntimePolicyBinding(model.model_id, options)


def _require_independent_evaluation_pool(previous_pool: Any, receiving_pool: Any) -> None:
    """Pure identity boundary; live Parent custody must end independently."""
    if receiving_pool is previous_pool or receiving_pool.budget is previous_pool.budget:
        raise ContractError('live Parent handoff requires an independent evaluation pool and account')


_EVALUATION_DATA_ISSUER = object()


class RustEvaluationData:
    """Independent Fresh-provider/FD/reference custody exported by a real Parent."""
    pool: Any
    parent_budget: ArtifactPhaseAdmission
    job: Any
    _parent: Any
    ownership: Any
    directory: Path
    deadline: float
    state_sha256: str
    binding: Json
    _failure_traceback: Any

    def __init__(self, issuer: object) -> None:
        if issuer is not _EVALUATION_DATA_ISSUER:
            raise ContractError('actual qualified Parent evaluation handoff required')
        self._cleanup_children: list[Any] = []
        self._loans: list[Any] = []
        self._state = self._metadata = b''
        self.provider: Any = None
        self.originals: _RetainedInferenceFiles | None = None
        self.scan: _FundedEvaluationScan | None = None
        self.reference: Any = None
        self.release_reference: Any = None
        self._closed, self._exposed, self._failed = False, False, False
        self._handoff_state = 'PREPARING'
        self._reference_started = False
        self._pending_loan: Any = None

    @classmethod
    def from_parent(cls, parent: Any, binding: Any, label: str, *, evaluation_job: Any,
                    pool: Any, parent_budget: ArtifactPhaseAdmission, pin: str) -> RustEvaluationData:
        import weakref

        from gf_common.domain import RuntimePolicyBinding
        from gf_common.rust_player import deadline_seconds
        from gf_training.group_pack_pool import PackedArtifactPool
        from gf_training.group_pack_resources import remaining
        from gf_training.hpo_gpu_lease import GpuJob
        from gf_training.rust_player_completion import ColdFinalizedRustParent
        if (type(parent) is not ColdFinalizedRustParent or type(evaluation_job) is not GpuJob
                or type(pool) is not PackedArtifactPool or type(parent_budget) is not ArtifactPhaseAdmission
                or parent_budget.source is not pool.budget or type(binding) is not RuntimePolicyBinding):
            raise ContractError('actual qualified Parent/job/binding/evaluation pool required')
        parent.require()
        _require_independent_evaluation_pool(parent._invocation.pool, pool)
        evaluation_job._require_open()
        admission, _, _ = parent._invocation.cold.require_pair()
        context = evaluation_job._admission['context']
        if (context['kind'] != 'evaluation' or context['run_id'] != admission['run_id']
                or context['campaign_id'] != admission['context']['campaign_id'] or evaluation_job.ticket != pin):
            raise ContractError('actual independent evaluation job differs from Parent/campaign/pin')
        output = cls(_EVALUATION_DATA_ISSUER)
        output.pool, output.parent_budget, output.job = pool, parent_budget, evaluation_job
        output._parent = weakref.ref(parent)
        output.ownership = parent._invocation._ownership
        output.directory = parent._invocation._parent_directory
        output.deadline = min(pool.deadline, deadline_seconds(admission['deadlines']['finish_bits']))
        parent._evaluation_exports.append(output)  # retain before any admission/IO
        try:
            remaining(output.deadline)
            pool.hold_close(output)
            output._pending_loan = parent_budget.reserve(32 * (len(parent._state) + len(parent._metadata) + len(parent._files)) + 65536)
            output._loans.append(output._pending_loan)
            output._pending_loan = None
            state, metadata = parent.state, parent.metadata
            validate_parent_binding(metadata, state, run_path=str(output.directory))
            output._state, output._metadata = canonical(state).encode(), canonical(metadata).encode()
            output.state_sha256 = metadata['state_sha256']
            if final_policy_binding(state, metadata) != (label, binding):
                raise ContractError('evaluation request differs from actual final Parent binding')
            ref = {'run_id': state['run_id'], 'token': pin, 'generation': state['updates'],
                'inference_sha256': metadata['inference_sha256'], 'generation_sha256': metadata['generation_sha256']}
            validate_reference_binding(ref, state, {'generation': state['updates'], 'inference_sha256': metadata['inference_sha256']},
                generation_sha256=metadata['generation_sha256'])
            output.binding = dict(ref)
            output._reference_started = True
            output.reference = _reference_transition(output.ownership, ref, 'acquire', state_sha256=output.state_sha256,
                pool=pool, parent_budget=parent_budget, deadline=output.deadline)
            directory = output.directory / 'artifacts' / f"{state['updates']:06}"
            output.originals = _RetainedInferenceFiles(directory, parent.files, budget=pool.budget,
                parent_budget=parent_budget, deadline=output.deadline)
            output._prepare_provider(state, binding, label)
            parent.require()
            remaining(output.deadline)
            output.originals._handoff_deadline()
            output.deadline = pool.deadline
            output._handoff_state, output._exposed = 'READY', True
            parent._evaluation_exports.remove(output)
            return output
        except BaseException as error:
            output._failed = True
            if output._reference_started:
                parent._failed = True
            output._handoff_state = 'FAILED'
            _capture_failure_graph(output, error)
            raise

    @classmethod
    def from_completed(cls, audit: RustCompletedParentAudit, binding: Any, label: str, *,
                       evaluation_job: Any, pool: Any, parent_budget: ArtifactPhaseAdmission,
                       pin: str) -> RustEvaluationData:
        from gf_training.group_pack_pool import PackedArtifactPool
        from gf_training.hpo_gpu_lease import GpuJob
        if (type(audit) is not RustCompletedParentAudit or type(evaluation_job) is not GpuJob
                or type(pool) is not PackedArtifactPool or pool is not audit.pool
                or parent_budget is not audit.parent_budget):
            raise ContractError('actual completed DATA owner/SAME evaluation job account required')
        audit.require()
        evaluation_job._require_open()
        context = evaluation_job._admission['context']
        if (audit._phase != 'READY' or audit.originals is None
                or context['kind'] != 'evaluation' or context['campaign_id'] != audit.campaign_id
                or context['run_id'] != audit.state['run_id'] or evaluation_job.ticket != pin
                or final_policy_binding(audit.state, audit.metadata) != (label, binding)):
            raise ContractError('actual complete import/final binding/evaluation context required')
        output = cls(_EVALUATION_DATA_ISSUER)
        output.pool, output.parent_budget, output.job = pool, parent_budget, evaluation_job
        output._parent = lambda: None
        output.ownership, output.directory = audit.ownership, audit.ownership.directory
        output.deadline = min(audit.deadline, pool.deadline)
        output.state_sha256 = audit.metadata['state_sha256']
        try:
            pool.hold_close(output)
            output._pending_loan = parent_budget.reserve(32 * (json_bound(audit.state) + json_bound(audit.metadata)) + 65536)
            output._loans.append(output._pending_loan)
            output._pending_loan = None
            output._state, output._metadata = canonical(audit.state).encode(), canonical(audit.metadata).encode()
            ref = {'run_id': audit.state['run_id'], 'token': pin, 'generation': audit.state['updates'],
                'inference_sha256': audit.metadata['inference_sha256'], 'generation_sha256': audit.metadata['generation_sha256']}
            output.binding = ref
            output._reference_started = True
            output.reference = _reference_transition(audit.ownership, ref, 'acquire', state_sha256=output.state_sha256,
                pool=pool, parent_budget=parent_budget, deadline=output.deadline)
            output.originals, audit.originals = audit.originals, None
            audit._children.remove(output.originals)
            output._prepare_provider(audit.state, binding, label)
            audit.require()
            output.originals._handoff_deadline()
            output.deadline, output._handoff_state, output._exposed = pool.deadline, 'READY', True
            audit._phase = 'TRANSFERRED'
            return output
        except BaseException as error:
            output._failed, audit._failed = True, True
            _capture_failure_graph(output, error)
            raise

    def _prepare_provider(self, state: Json, binding: Any, label: str) -> None:
        from gf_common.card_instance import action_config_v3
        from gf_common.card_instance_rules import card_instance_rules
        from gf_models.card_instance_artifacts import RetainedGameArtifacts
        assert self.originals is not None
        limit = min(self.parent_budget.limit - self.parent_budget.used, self.pool.budget.limit - self.pool.budget.used)
        self.scan = _FundedEvaluationScan(self.pool, self.parent_budget, max_owned_bytes=limit, deadline=self.deadline)
        self.provider = RetainedGameArtifacts.for_hpo_registry({label: binding}, action_config_v3(state['effective_config']['actions']),
            pool_size=1, rules=card_instance_rules(), max_owned_bytes=limit, preflight_deadline=self.deadline,
            finalization_deadline=self.pool.deadline, _scan=self.scan)
        actual = self.provider.journal_metadata(label)
        directory = self.directory / 'artifacts' / f"{state['updates']:06}"
        if ({row['path']: row['sha256'] for row in actual['files']} != self.originals.files
                or actual['policy_options'] != binding.policy_options
                or actual['policy_options']['checkpoint'] != str(directory)):
            raise ContractError('actual Fresh provider differs from same held inference originals')
        self.scan.settle()
        self.originals.require()

    def require(self) -> None:
        try:
            if self._closed or self._failed or not self._exposed or self._handoff_state != 'READY':
                raise ContractError('actual positively handed-off evaluation DATA required')
            self.job._require_open()
            self.pool.check()
            assert self.originals is not None
            self.originals.require()
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def finalize(self, *, deadline: float) -> Json:
        self.require()
        try:
            result = self.provider.finalize(deadline=min(deadline, self.deadline))
            assert self.scan is not None
            self.scan.settle()
            assert self.originals is not None
            self.originals.require()
            return result
        except BaseException as error:
            self._failed = True
            _capture_failure_graph(self, error)
            raise

    def close(self, *, ownership: Any = None) -> None:
        if self._closed:
            return
        if self._exposed:
            receipt = self.job._receipt
            if receipt is None:
                raise ContractError('actual evaluation worker retirement receipt required before DATA close')
            self.job.require_confirmed_termination(receipt)
        _clear_failure_frames(self)
        if self.provider is not None:
            self.provider.release()
            self.provider = None
        close_resources(self, [])
        if self.originals is not None:
            self.originals.close()
            self.originals = None
        if self.scan is not None:
            self.scan.close()
            self.scan = None
        if self.reference is not None:
            from contextlib import ExitStack

            from gf_training.lineage import run_lock
            with ExitStack() as scope:
                actual = ownership
                if actual is None:
                    actual = (self.ownership if not self._exposed else scope.enter_context(run_lock(self.directory)))
                self.release_reference = _reference_transition(actual, self.binding, 'release', state_sha256=self.state_sha256,
                    pool=self.pool, parent_budget=self.parent_budget, deadline=self.deadline,
                    acquisition_sha256=self.reference.receipt_sha256)
            self.reference = None
        elif self._reference_started:
            raise ContractError('uncertain reference acquisition remains quarantined')
        self._state = self._metadata = b''
        if self._pending_loan is not None:
            self._pending_loan.close()
            self._pending_loan = None
        for loan in self._loans:
            loan.close()
        self._loans.clear()
        for hold in tuple(self.pool._close_holds):
            if hold.owner is self:
                hold.retire(self)
        parent = self._parent()
        if parent is not None and self in parent._evaluation_exports:
            parent._evaluation_exports.remove(self)
        self._handoff_state, self._closed = 'CLOSED', True
        _discard_failure_graph(self)

    def __reduce__(self) -> Any:
        raise TypeError('actual evaluation DATA/ref custody cannot be copied or serialized')


class _RetainedEvaluationMetadata:
    """Fresh copied result DATA, borrowed only until its actual audit last use."""
    def __init__(self, manifest: Json, proof: Json, *, pool: Any,
                 parent_budget: ArtifactPhaseAdmission) -> None:
        from gf_training.group_pack_pool import PackedArtifactPool
        if type(pool) is not PackedArtifactPool or type(parent_budget) is not ArtifactPhaseAdmission or parent_budget.source is not pool.budget:
            raise ContractError('actual SAME result metadata pool/phase required')
        pool.check()
        self._loan: Any = None
        self._value: Json = {}
        self._bytes = b''
        self._closed = False
        self._cleanup_children: list[Any] = []
        try:
            self._loan = parent_budget.reserve(32 * json_bound((manifest, proof)) + 65536)
            self._bytes = canonical({'manifest': manifest, 'proof': proof}).encode()
            self._value = json.loads(self._bytes)
        except BaseException as error:
            _capture_failure_graph(self, error)
            raise

    @property
    def serialized_bytes(self) -> int:
        return len(self._bytes)

    @property
    def quota_bytes(self) -> int:
        """Existing32x JSON+64KiB estimate; not measured Python heap/RSS."""
        return 0 if self._loan is None else int(self._loan.size)

    @property
    def value(self) -> Json:
        if self._closed:
            raise ContractError('returned evaluation metadata lifetime ended')
        return self._value

    def close(self) -> None:
        if self._closed:
            return
        _clear_failure_frames(self)
        close_resources(self, [])
        self._value.clear()
        self._bytes = b''
        if self._loan is not None:
            self._loan.close()
            self._loan = None
        self._closed = True
        _discard_failure_graph(self)

    def __reduce__(self) -> Any:
        raise TypeError('actual evaluation metadata allocation ownership cannot be serialized')


class _BorrowedVerifierRetirement:
    """Actual readonly wrapper and outer hold; no Source/Parent surrogate."""
    def __init__(self, verifier: Any, hold: Any, owner: object) -> None:
        from gf_training.group_pack_game_source import PackedSourceOwner
        from gf_training.group_pack_pool import PoolCloseHold
        if type(verifier) is not PackedSourceOwner or type(hold) is not PoolCloseHold or hold.owner is not owner:
            raise ContractError('actual verifier/registered outer hold/SAME owner required')
        self.verifier, self.hold, self.owner = verifier, hold, owner
        self._cleanup_children: list[Any] = []
        self._traceback: Any = None
        self.closed = False

    def capture(self, error: BaseException) -> None:
        _capture_failure_graph(self, error, exclude=(self.verifier, self.owner))

    def close(self) -> None:
        if self.closed:
            return
        try:
            _clear_failure_frames(self)
            close_resources(self, [])
            self.verifier.close(_audit_hold=self.hold, _audit_owner=self.owner)
        except BaseException as error:
            self.capture(error)
            raise
        self.closed = True
        self._traceback = None
        _discard_failure_graph(self)

    def __reduce__(self) -> Any:
        raise TypeError('actual verifier/hold retirement custody cannot be serialized')
