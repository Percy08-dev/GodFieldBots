"""Actual training invocation pool, durable cache pins and postcommit weight sync."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, digest
from gf_common.hpo_resident import CACHE_ENTRIES, CACHE_WEIGHT_BYTES
from gf_models.sharded_arrays import read_json
from gf_simulator_client.resident_pool import ResidentParticipantPool
from gf_training.hpo_resident import ResidentPoolCustody
from gf_training.hpo_retention import pin_hpo_cache, release_hpo_cache


class ResidentTrainingRuntime:
    """Never outlives the SAME actual job, original parent lock or invocation."""
    def __init__(self, job: Any, ownership: Any, config: Any, state: Json, *, deadline: float, cleanup_deadline: float | None = None, owned_registry: Any = None) -> None:
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_models.structured_runtime import hpo_resident_training_admission
        from gf_training.hpo_lineage import require_hpo_run_owner
        require_hpo_run_owner(ownership.directory, ownership)
        if state['run_id'] != job._admission['context']['run_id']:
            raise ContractError('resident runtime differs from actual training invocation')
        self.owned_registry=owned_registry
        self.job, self.ownership, self.config = job, ownership, config
        self.directory, self.state = ownership.directory, state
        self.deadline = deadline if cleanup_deadline is None else cleanup_deadline
        self.limit = config.hpo_spec['operating']['parent_budget_bytes']
        envelope = hpo_resident_training_admission(config.model.architecture(layout=layout), config.model.inference,
            config.hpo_spec['training_recipe']['tbptt_length'], config.hpo_spec['operating'],
            config.hpo_spec['runtime'], layout=layout)
        self.custody = ResidentPoolCustody(job, run_id=state['run_id'], pool_id=uuid.uuid4().hex)
        self.pool = ResidentParticipantPool(self.custody, resident_bytes=envelope['worker_bytes']//envelope['workers'],
            deadline=self.deadline, acquire_cache=self.acquire_cache, release_cache=self.release_cache,
            owned_snapshots=owned_registry is not None)
        self.pool.execution_deadline = deadline
        self.entries: dict[int, OrderedDict[str, tuple[int, Any]]] = {}
        self.pins: dict[str,tuple[Any,Any]] = {}
        self.acquisitions: dict[str,tuple[Any,Any]] = {}
        self.transition: Any = None
        self.envelope = envelope

    def _identity(self, owner: Any) -> Json:
        return self.custody._identity(self.custody._row(owner.lease))

    def _acquire(self,owner: Any,generation: int,inference_sha256: str) -> Any:
        attempt=self.custody.begin_cache_acquisition(owner.lease,uuid.uuid4().hex,generation,inference_sha256)
        self.acquisitions[attempt.token]=owner,attempt  # Before fallible transaction/intent publication.
        try:
            receipt=(self.owned_registry.pin_cache(attempt,deadline=self.pool.execution_deadline) if self.owned_registry is not None else
                pin_hpo_cache(self.directory,attempt.token,generation,inference_sha256,
                owner=attempt.owner_identity,max_owned_bytes=self.limit,deadline=self.pool.execution_deadline,ownership=self.ownership))
            attempt.receipt=receipt  # Retain before any fallible custody observation/map insertion.
            self.pins[attempt.token]=owner,receipt
            self.custody.observe_cache_pin(receipt)
            return receipt
        except BaseException:
            self.custody.fail()
            raise

    def _reconcile_after_reap(self,owner: Any,attempt: Any) -> Any:
        self.custody.require_acquisition_reap(attempt)
        if self.acquisitions.get(attempt.token)!=(owner,attempt):
            raise ContractError('cache reconciliation requires SAME original acquisition attempt')
        if attempt.receipt is None:
            # Retry only this original idempotent acquire under the original
            # held parent owner, after actual process/I/O/setup retirement.
            attempt.receipt=(self.owned_registry.pin_cache(attempt,deadline=self.deadline) if self.owned_registry is not None else
                pin_hpo_cache(self.directory,attempt.token,attempt.generation,attempt.inference_sha256,
                owner=attempt.owner_identity,max_owned_bytes=self.limit,deadline=self.deadline,ownership=self.ownership))
        self.pins[attempt.token]=owner,attempt.receipt
        return attempt.receipt

    def _pin(self, owner: Any, binding: Any) -> Any:
        options = binding.policy_options
        if options is None or binding.model_id == 'heuristic_refresh':
            return None
        path = Path(options['checkpoint'])
        if path.parent != self.directory/'artifacts' or not path.name.isdecimal():
            raise ContractError('resident training cache escaped original parent artifacts')
        return self._acquire(owner,int(path.name),options['expected_manifest_digest'])

    async def acquire_cache(self, owner: Any, binding: Any) -> str:
        entries = self.entries.setdefault(owner.lease.key, OrderedDict())
        policy_sha = digest(asdict(binding))
        options = binding.policy_options
        # Existing immutable binding can be recognized without reading files.
        existing = next((identity for identity, (_, receipt) in entries.items()
            if receipt is not None and receipt.binding['policy_sha256'] == policy_sha), None)
        if existing is not None:
            entries.move_to_end(existing)
            return existing
        receipt = self._pin(owner, binding)
        if receipt is None:
            identity, size = policy_sha, 0
        else:
            identity = receipt.binding['cache_identity']
            # Durable pin acquisition precedes this first direct checkpoint read.
            manifest = (self.owned_registry.snapshots[receipt.binding['generation']].inference_manifest
                if self.owned_registry is not None else
                read_json(Path(options['checkpoint'])/'inference.json', expected_digest=options['expected_manifest_digest']))
            size = manifest['weights']['decoded_bytes']
            if digest(manifest['weights']['files']) != receipt.binding['weights_sha256']:
                raise ContractError('resident actual checkpoint differs from acquired cache reference')
        if identity in entries:
            entries.move_to_end(identity)
            return identity
        if size > CACHE_WEIGHT_BYTES:
            raise ContractError('resident cache single checkpoint exceeds weight bound')
        while len(entries) >= CACHE_ENTRIES or sum(row[0] for row in entries.values())+size > CACHE_WEIGHT_BYTES:
            candidates = [(key,value) for key,value in entries.items() if key != owner.protected_identity]
            if not candidates:
                raise ContractError('resident cache capacity cannot evict current identity')
            previous, (_, old) = candidates[0]
            checkpoint = old.binding['inference_sha256'] if old is not None else previous
            await owner.control('EVICT', checkpoint=checkpoint, cache_identity=previous,
                timeout=max(0., min(10., self.deadline-time.monotonic())))
            self.release_cache(owner, previous)
            del entries[previous]
        entries[identity] = size, receipt
        return identity

    def release_cache(self, owner: Any, identity: str | None) -> None:
        entries: Any = self.entries.get(owner.lease.key, {})
        selected=([attempt for original,attempt in self.acquisitions.values() if original is owner] if identity is None
            else ([] if entries[identity][1] is None else [self.acquisitions[entries[identity][1].binding['token']][1]]))
        for attempt in tuple(selected):
            receipt=self._reconcile_after_reap(owner,attempt) if identity is None else attempt.receipt
            authority=(self.custody.cache_release_acquisition_after_reap(attempt,receipt) if identity is None
                else self.custody.cache_release(receipt.binding))
            if self.owned_registry is not None:
                self.owned_registry.release_cache(receipt.binding,authority=authority,deadline=self.deadline)
            else:
                release_hpo_cache(self.directory,receipt.binding,authority=authority,
                    max_owned_bytes=self.limit,deadline=self.deadline,ownership=self.ownership)
            self.pins.pop(attempt.token,None)
            self.acquisitions.pop(attempt.token)
        if identity is None:
            entries.clear()

    async def after_commit(self, generation: int, record: Json) -> None:
        """Caller has retired Source/run/registry snapshots after durable append."""
        from gf_common.card_instance_rules import card_instance_rules
        from gf_simulator_client.card_instance_episode import episode_initial
        from gf_training.hpo_source import hpo_binding
        from gf_training.lineage import checkpoint_path
        authority = self.pool.authority
        authority.require(self.job.ledger)
        previous = generation-1
        affected = [owner for owner in self.pool.owners if not owner.lease.released and any(
            receipt is not None and receipt.binding['generation'] == previous
            for _, receipt in self.entries.get(owner.lease.key, {}).values())]
        if not affected:
            return
        # Acquire all durable references before constructing bindings/reading weights.
        pins = []
        for owner in affected:
            receipt=self._acquire(owner,generation,record['inference_sha256'])
            pins.append((owner, receipt))
        _, binding = (self.owned_registry.binding(generation) if self.owned_registry is not None else
            hpo_binding(self.directory, self.state, record, generation))
        manifest=(self.owned_registry.snapshots[generation].inference_manifest if self.owned_registry is not None else
            read_json(checkpoint_path(self.directory,generation)/'inference.json',expected_digest=record['inference_sha256']))
        size=manifest['weights']['decoded_bytes']
        self.transition = self.job.ledger.reserve_resident_transition(self.envelope['staging_bytes'], authority=authority,
            target_generation=generation, target_checkpoint_sha256=record['inference_sha256'],
            affected_workers=tuple(owner.lease for owner in affected))
        with self.transition:
            for owner, receipt in pins:
                entries = self.entries.setdefault(owner.lease.key, OrderedDict())
                # Explicit eviction is ACK-backed; protect all separately frozen identities.
                while len(entries) >= CACHE_ENTRIES or sum(row[0] for row in entries.values())+size > CACHE_WEIGHT_BYTES:
                    candidates = [(key,value) for key,value in entries.items() if key != owner.protected_identity]
                    if not candidates:
                        raise ContractError('resident sync capacity cannot evict current identity')
                    identity, (_, old) = candidates[0]
                    await owner.control('EVICT', checkpoint=old.binding['inference_sha256'] if old is not None else identity,
                        cache_identity=identity, timeout=10.)
                    self.release_cache(owner, identity)
                    del entries[identity]
                entries[receipt.binding['cache_identity']] = size, receipt
                initial = episode_initial(binding,0,self.config.actions,card_instance_rules())
                await owner.control('SYNC', checkpoint=record['inference_sha256'], generation=generation,
                    payload=initial, timeout=10.)
                owner.protected_identity = receipt.binding['cache_identity']
                await owner.control('PARK', checkpoint=record['inference_sha256'], timeout=10.)
        self.transition = None

    def require_closed(self) -> None:
        self.custody.require_resource_retirement(self.job,self.job.ledger)
        if not self.pool.closed or self.pins or self.acquisitions or self.transition is not None:
            raise ContractError('resident original pool/cache references/transition retirement unconfirmed')

    async def close(self) -> None:
        primary = None
        try:
            await self.pool.close()
        except BaseException as error:  # noqa: BLE001 - verify actual facts even after a failed CLOSE
            primary = error
        try:
            if self.transition is not None and self.transition.active:
                self.transition.retire_failed_transition()
                self.transition = None
            self.require_closed()
            if self.owned_registry is not None:
                for snapshot in tuple(self.pool._owned_snapshots.values()):
                    self.pool.unbind_owned_snapshot(snapshot)
        except BaseException as error:  # noqa: BLE001 - retain the concrete original cleanup owner
            failed = primary or error
            vars(failed)['cleanup_owner'] = self
            vars(failed)['_resident_cleanup_owner'] = self
            if primary is not None:
                primary.add_note('resident positive retirement unavailable: '+str(error))
            raise failed
        if primary is not None:
            raise primary


class ResidentEvaluationRuntime:
    """One panel; cache uses the genuine outer evaluation pins through actual reap."""
    def __init__(self, job: Any, plan: Any, artifacts: Any, config: Any) -> None:
        from gf_common.card_instance_rules import card_instance_rules
        from gf_models.structured_runtime import own_game_worker_settings_v3
        from gf_training.hpo_resident import EvaluationPoolScope
        scope = EvaluationPoolScope(job,plan,artifacts,plan._pins,
            max_owned_bytes=config.hpo_spec['operating']['source_budget_bytes'],deadline=plan.execution_deadline)
        settings = [own_game_worker_settings_v3(binding.model_id,config.actions,binding.policy_options,rules=card_instance_rules())
            for binding in plan.bindings.values()]
        residency = max(row[2] for row in settings)+CACHE_WEIGHT_BYTES
        temporary = max(row[3] for row in settings)
        if config.run.concurrency*2*residency+config.run.concurrency*temporary > job.ledger.global_cuda_bytes:
            raise ContractError('resident evaluation cache/context/activation exceeds managed CUDA budget')
        self.plan,self.artifacts,self.deadline = plan,artifacts,plan.work_deadline
        self.custody = ResidentPoolCustody(job,run_id=plan.linkage['evaluation_id'],pool_id=uuid.uuid4().hex,evaluation_scope=scope)
        self.entries: dict[int, OrderedDict[str,tuple[int,str]]] = {}
        self.pool = ResidentParticipantPool(self.custody,resident_bytes=residency,deadline=plan.work_deadline,
            acquire_cache=self.acquire_cache)
        self.pool.admit(plan)

    async def acquire_cache(self, owner: Any, binding: Any) -> str:
        scope = self.custody.evaluation_scope
        if scope is None:
            raise ContractError("evaluation runtime lost original scope")
        scope.require(self.custody.job,self.plan)
        entries = self.entries.setdefault(owner.lease.key,OrderedDict())
        policy_sha = digest(asdict(binding))
        options = binding.policy_options
        if options is None:
            identity, size, checkpoint = policy_sha,0,policy_sha
        else:
            self.artifacts.require_options(binding.model_id,options)
            self.artifacts.finalize(deadline=self.deadline)
            manifest = read_json(Path(options['checkpoint'])/'inference.json',expected_digest=options['expected_manifest_digest'])
            size,checkpoint = manifest['weights']['decoded_bytes'],options['expected_manifest_digest']
            identity = digest({'policy_sha256':policy_sha,'inference_sha256':checkpoint,
                'weights_sha256':digest(manifest['weights']['files'])})
        if identity in entries:
            entries.move_to_end(identity)
            return identity
        if size > CACHE_WEIGHT_BYTES:
            raise ContractError('resident evaluation cache exceeds single weight bound')
        while len(entries)>=CACHE_ENTRIES or sum(row[0] for row in entries.values())+size>CACHE_WEIGHT_BYTES:
            candidates = [(key,value) for key,value in entries.items() if key != owner.protected_identity]
            if not candidates:
                raise ContractError('resident evaluation cannot evict current cache identity')
            previous,(_,old_checkpoint) = candidates[0]
            await owner.control('EVICT',checkpoint=old_checkpoint,cache_identity=previous,timeout=10.)
            del entries[previous]
        # Scope is still the SAME actual immutable parent-set/evaluation pins;
        # this is an admitted load intent, never a fabricated training cache pin.
        self.custody.observe_evaluation_cache(owner.lease,self.plan,binding,identity)
        entries[identity] = size,checkpoint
        return identity

    async def close(self) -> None:
        await self.pool.close()
        self.custody.require_resource_retirement(self.custody.job,self.custody.ledger)
