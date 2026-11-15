"""One invocation's retained original evidence before collection allocation.

This is custody of the complete historical validator's actual scan, never a
cache or Source/update authority. The caller keeps all live validation guards.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, digest
from gf_training.structured_lineage import _ChainScan


class _PrecollectionValidation:
    phase: str
    scan: _ChainScan | None
    admission: Any
    online: Any
    plan: Any
    ledger: Any
    cleanup_deadline: float
    cap: int
    deadline: float
    ownership: Any
    process: Any
    job: Any
    runtime: Any
    owner_state: Any
    pid: int
    thread: int
    directory: Path
    output: Path
    original_plan: str
    facts: Any
    facts_digest: str | None
    native: Any
    native_identity: Any
    admission_epoch: Any
    scratch_bytes: int

    @classmethod
    def attach(cls, online: Any, plan: Any, ledger: Any, *, max_owned_bytes: int,
               deadline: float, cleanup_deadline: float) -> _PrecollectionValidation:
        from gf_training.card_instance_source import _OnlineGameOwner
        from gf_training.hpo_resident_runtime import ResidentTrainingRuntime
        if (type(online) is not _OnlineGameOwner or type(online.resident_runtime) is not ResidentTrainingRuntime
                or online.precollection_validation is not None or online.ledger is not ledger):
            raise ContractError('precollection requires SAME unoccupied actual resident owner')
        value = cls.__new__(cls)
        value.phase, value.scan, value.admission = 'new', None, None
        value.online, value.plan, value.ledger = online, plan, ledger
        value.cleanup_deadline = cleanup_deadline
        online.precollection_validation = value  # Before any fallible allocation/admission.
        try:
            value.cap, value.deadline = max_owned_bytes, deadline
            value.ownership, value.process, value.job = online.ownership, online.process, online.hpo_job
            value.runtime, value.owner_state = online.resident_runtime, online.state
            value.pid, value.thread = os.getpid(), threading.get_ident()
            value.directory, value.output = online.ownership.directory, plan.output.absolute()
            value.scan = _ChainScan(directory=value.directory, max_owned_bytes=max_owned_bytes, deadline=deadline)
            value.scratch_bytes = 0
            value.scan.charge(16384)  # Carrier, native identity, fingerprints and their bookkeeping.
            value.original_plan = value._plan_digest()
            value.facts = None
            value.facts_digest = None
            value.native = None
            value.native_identity = None
            value.admission_epoch = None
        except BaseException:
            value.phase = 'failed'
            raise
        return value

    def _reserve_scratch(self, size: int) -> None:
        assert self.scan is not None
        self.scan.check()
        if size > self.scratch_bytes:
            # Keep the real high-water reservation until terminal close. Unlike
            # per-recheck temporary refunds, shared control admission records
            # exact effect evidence even if acquisition/refund raises.
            self.scan.charge(size - self.scratch_bytes)
            self.scratch_bytes = size

    def _fingerprint(self, value: Any) -> str:
        from gf_training.structured_rollout import _owned_bytes
        assert self.scan is not None
        # Admit encoding/sorting/hash scratch before allocating it. The decoded
        # originals are already charged by SealedScan.object, not copied here.
        self._reserve_scratch(65536 + 8 * _owned_bytes(value))
        return digest(value)

    def _plan_digest(self) -> str:
        from gf_training.hpo_resident import _plan_identity
        from gf_training.structured_rollout import _owned_bytes
        assert self.scan is not None
        self._reserve_scratch(65536 + 8 * (_owned_bytes(self.plan.groups) + _owned_bytes(self.plan.bindings)
            + _owned_bytes(self.plan.linkage) + len(str(self.plan.output))))
        return self._fingerprint({**_plan_identity(self.plan), 'kind':self.plan.kind,
            'started_monotonic':self.plan.started_monotonic})

    def _require(self, online: Any, plan: Any, ledger: Any) -> None:
        if (type(self) is not _PrecollectionValidation or online is not self.online
                or online.precollection_validation is not self or plan is not self.plan or ledger is not self.ledger
                or online.ownership is not self.ownership or online.process is not self.process
                or online.hpo_job is not self.job or online.resident_runtime is not self.runtime
                or online.state is not self.owner_state or online.ledger is not ledger
                or online.ownership.directory != self.directory or self.process.online_owner is not online
                or self.pid != os.getpid() or self.thread != threading.get_ident()
                or self.phase not in ('new', 'ready') or self.scan is None or self.scan._closed
                or self.output != plan.output.absolute()):
            raise ContractError('precollection original owner/plan/process/job/lifecycle differs')
        self.scan.check()
        if time.monotonic() >= self.deadline:
            raise TimeoutError('precollection original deadline')
        if self.output.exists() or self.output.is_symlink() or self._plan_digest() != self.original_plan:
            raise ContractError('precollection plan changed or output already exists')

    def _native(self, *, initial: bool = False) -> None:
        from gf_common.preprocessing import preparation_config
        from gf_common.training_opponents import training_opponents
        if training_opponents(self.owner_state['effective_config']) is not None:
            from gf_training.native_backend import require_native_refresh_opponents
            # Mixed HPO always requires the current native opponent capability,
            # including Python preprocessing, just like full parent validation.
            self._reserve_scratch(65536)
            require_native_refresh_opponents()
        settings = preparation_config(self.owner_state['effective_config']['preprocessing'])
        if settings.backend != 'rust':
            return
        from gf_training.native_backend import (
            binary_identity,
            hpo_reader_settings,
            load_native,
            require_ppo_model,
        )
        assert self.scan is not None
        # Charge identity/build encoding metadata before native helpers allocate.
        self._reserve_scratch(65536)
        module = load_native()
        hpo_reader_settings(settings, module, packed='storage_binding' in self.owner_state, resident=True)
        require_ppo_model(self.owner_state['model_id'], module)
        identity = binary_identity(module)
        if initial:
            self.native, self.native_identity = module, identity
            self.scan.charge(65536)  # Retained binary/build identity and path.
            with self.scan.file(Path(identity['extension_path']), identity['extension_sha256']):
                pass
        elif module is not self.native or self._fingerprint(identity) != self._fingerprint(self.native_identity):
            raise ContractError('precollection loaded native module/binary identity changed')

    @property
    def owned_bytes(self) -> int:
        assert self.scan is not None
        pool = self.scan._packed_pool
        # Shared pool usage already contains admitted control charges.
        if self.phase == 'closed':
            return 0
        return self.scan.retained if pool is None else pool.budget.used + self.scan.retained - self.scan._pool_control_charge

    def history(self, online: Any, plan: Any, ledger: Any, *, resident_phase: Any,
                max_owned_bytes: int, deadline: float) -> Any:
        from gf_training.card_instance_source import recheck_game_parent_snapshot
        from gf_training.hpo_resident import CollectionAdmission
        if self.phase == 'closed':
            raise ContractError('precollection expiry is terminal')
        try:
            self._require(online, plan, ledger)
            self.cap = min(self.cap, max_owned_bytes)
            assert self.scan is not None
            self.deadline = min(self.deadline, deadline)
            self.scan.deadline = min(self.scan.deadline, self.deadline)
            self.scan.limit = min(self.scan.limit, self.cap)
            self.scan.check()
            if type(resident_phase) is not CollectionAdmission:
                raise ContractError('precollection requires current original collection admission')
            resident_phase.require(plan, ledger)
            if self.phase == 'new':
                self.admission, self.admission_epoch = resident_phase, resident_phase.pool._epoch
                self.phase = 'building'
                self._native(initial=True)
                from gf_training.hpo_source import validate_hpo_parent
                facts = validate_hpo_parent(self.directory, require_complete=False,
                    max_owned_bytes=self.cap, deadline=self.deadline, _scan=self.scan)
                self._native()
                self.facts_digest = self._fingerprint(facts)
                self.facts = facts
                self.phase = 'ready'
            elif resident_phase is not self.admission or resident_phase.pool._epoch != self.admission_epoch:
                raise ContractError('precollection original admission or epoch changed')
            else:
                self._native()
                recheck_game_parent_snapshot(self.scan)
                if self._fingerprint(self.facts) != self.facts_digest:
                    raise ContractError('precollection checked historical facts changed')
            if self.owned_bytes > self.cap:
                raise ContractError('precollection actual Source reservation exceeds original cap')
            # Existing resident update admission includes parent + learner +
            # resident workers + Source (+ numeric). No collector pool exists
            # during this overlap, and its registry retains the parent envelope.
            op = self.runtime.config.hpo_spec['operating']
            learner = 0 if ledger.learner is None else ledger.learner.resident_bytes
            workers = sum(lease.resident_bytes for lease in ledger.workers.values())
            overlap = op['parent_budget_bytes'] + learner + workers + self.owned_bytes
            if (overlap > self.runtime.envelope['ram_phases']['update']
                    or self.runtime.envelope['ram_phases']['update'] > op['managed_ram_bytes']):
                raise ContractError('precollection actual overlap exceeds original admitted update envelope')
            if self.scan._packed_pool is not None:
                self.scan._packed_pool.require_terminal_close()
            return self.facts
        except BaseException:
            self.phase = 'failed'
            raise

    def finish(self, online: Any, plan: Any, ledger: Any) -> None:
        self._require(online, plan, ledger)
        if self.phase != 'ready':
            raise ContractError('precollection cannot expire incomplete validation')
        # All current caller guards have succeeded; one last still-owned sweep
        # precedes actual close and the caller's mkdir/collector allocation.
        self.history(online, plan, ledger, resident_phase=self.admission,
            max_owned_bytes=self.cap, deadline=self.deadline)
        self.close()

    def close(self) -> None:
        if self.phase == 'closed':
            return
        self.phase = 'closing'
        try:
            # Terminal close performs no work or learner-liveness checks and
            # never extends work deadlines. The original cleanup allowance bounds it.
            if self.scan is not None:
                self.scan.close(recheck=False)
                if (not self.scan._closed or self.scan._packed_pool is not None
                        and (not self.scan._packed_pool.closed or self.scan._packed_pool.budget.used != 0)):
                    raise ContractError('precollection original scan/account retirement unconfirmed')
            if time.monotonic() >= self.cleanup_deadline:
                raise TimeoutError('precollection original cleanup allowance overrun')
            if self.online.precollection_validation is not self:
                raise ContractError('precollection original attachment changed during cleanup')
            self.phase = 'closed'
            self.online.precollection_validation = None
            self.facts = None
            self.native = self.native_identity = None
            self.admission = None
            if self.scan is not None:
                # Physical close is confirmed before dropping decoded metadata.
                # Keep only the closed shell for lifecycle/accounting inspection.
                self.scan.files.clear()
                self.scan.inventories.clear()
                self.scan.logical_files.clear()
                self.scan._packed_owners.clear()
                self.scan._packed_state = None
                self.scan._game_source_inventories.clear()
                self.scan._game_source_settings = None
                self.scan._hpo_reference_snapshot = None
                self.scan._hpo_cache_reference_snapshot = None
                self.scan.retained = 0
                self.scratch_bytes = 0
        except BaseException as error:
            self.phase = 'failed'
            error.__dict__['cleanup_owner'] = self
            error.__dict__['_hpo_precollection_cleanup_owner'] = self
            raise
