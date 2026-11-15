"""Fresh producer-owned operations in the shared PPO experiment loop."""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, digest
from gf_common.group_pack import StorageSettings
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_config import InferenceBindingV2
from gf_training.card_instance_source import _OnlineGameOwner
from gf_training.group_pack_resources import StorageBudget
from gf_training.hpo_config import hpo_training_config
from gf_training.hpo_owned_lineage import OwnedControlWriter, OwnedRunRegistry
from gf_training.hpo_ppo import _HpoOperations
from gf_training.lineage import require_hpo_run_owner


class _OwnedHpoOperations(_HpoOperations):
    registry_schema=6

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args,**kwargs)
        self.owned_registry: OwnedRunRegistry | None=None
        self.writer: OwnedControlWriter | None=None
        self.budget: Any=None
        self.transaction: Any=None
        self.last_snapshot: Any=None
        self.numeric: Any=None
        self._startup_child: Any=None
        self._startup_start_requested=False

    def validate_config(self, config: Any) -> Any:
        if (config.hpo_spec.get('schema_version')!=4 or config.hpo_spec.get('origin')!='producer-owned-v1'
                or config.hpo_spec.get('encoder')!='canonical-python' or config.preprocessing.backend!='python'):
            raise ContractError('owned operations require explicit spec4 canonical Python producer')
        return super().validate_config(config)

    def initialize_state(self, state: Json, config: Any) -> None:
        super().initialize_state(state,config)
        self.registry_schema=6
        state['run_variant']='hpo-ppo-owned-v1'

    def persist_control(self, path: Path, value: Json, config: Any, *, deadline: float, original: Any=None) -> None:
        if original is not None: raise ContractError('owned control cannot borrow a historical scanner')
        if self.writer is None:
            owner=require_hpo_run_owner(path.parent)
            self.budget=StorageBudget(StorageSettings(budget_bytes=self.cap(config)),deadline)
            self.writer=OwnedControlWriter(owner,budget=self.budget,limit=config.hpo_spec['operating']['parent_budget_bytes'])
        self.writer.write(path,value,deadline=deadline)

    def make_child(self, directory: Path, config: Any, state: Json, *, deadline: float,
                   cleanup_deadline: float | None=None) -> Any:
        from gf_training.hpo_gpu_lease import require_training_job
        from gf_training.hpo_owned_process import HpoOwnedLearningProcess
        from gf_training.hpo_resident_runtime import ResidentTrainingRuntime
        operating=config.hpo_spec['operating']
        require_training_job(self.job,ledger=self.job.ledger,run_id=state['run_id'],
            global_cuda_bytes=operating['managed_cuda_bytes'],max_workers=operating['max_participant_workers'])
        if self.writer is None: raise ContractError('owned initial controls require actual writer')
        self.budget.deadline=cleanup_deadline or deadline
        from gf_models.structured_sealed import SealedScan
        external=SealedScan(max_owned_bytes=operating['parent_budget_bytes'],deadline=deadline)
        with (self.writer.parent_budget.reserve(1024**2+65536),
              external.file(Path(config.server.config).absolute(),state['identity']['integration']['server_config_sha256'])):
            pass
        registry=OwnedRunRegistry.__new__(OwnedRunRegistry)
        self.owned_registry=registry
        OwnedRunRegistry.__init__(registry,self.ownership,state,budget=self.budget,
            parent_budget_bytes=operating['parent_budget_bytes'],writer=self.writer)
        child=HpoOwnedLearningProcess.__new__(HpoOwnedLearningProcess)
        self._startup_child=child
        HpoOwnedLearningProcess.__init__(child,directory,config.training.max_log_bytes,config.model.architecture(layout=LAYOUT),
            hpo_training_config(config.hpo_spec['training_recipe']),InferenceBindingV2(**state['execution']['learner']['binding']),
            digest(asdict(config.actions)),state['resume_identity'],actions=config.actions,ledger=self.job.ledger,
            cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'],cleanup_deadline=deadline,
            storage_binding=state.get('storage_binding'),phase_limits={key:operating[key] for key in ('numeric_budget_bytes','save_budget_bytes')},
            budget=self.budget,progress=self.learner_progress,diagnostics=self.diagnostics is not False)
        registry.operations=self
        registry.attach_process(child)
        registry.external_controls=external
        self.online=_OnlineGameOwner(self.ownership,state,child,self.job.ledger,hpo_job=self.job,owned_registry=registry)
        child.online_owner=self.online
        self.ledger=self.job.ledger
        self.resident_runtime=ResidentTrainingRuntime.__new__(ResidentTrainingRuntime)
        ResidentTrainingRuntime.__init__(self.resident_runtime,self.job,self.ownership,config,state,deadline=deadline,
            cleanup_deadline=cleanup_deadline,owned_registry=registry)
        self.resident_runtime.pool.progress=self.progress
        self.online.resident_runtime=self.resident_runtime
        return child,self.ledger

    async def abort_startup(self, error: BaseException, state: Json, config: Any, *, deadline: float) -> None:
        """Retire the exact provisional graph before make_child returns a child."""
        failures=[]
        runtime=self.resident_runtime
        if runtime is not None:
            try:
                pool=getattr(runtime,'pool',None)
                if pool is not None: await pool.close()
                custody=getattr(runtime,'custody',None)
                if custody is not None: custody.require_resource_retirement(self.job,self.job.ledger)
            except BaseException as cleanup:  # noqa: BLE001 - retain every provisional owner and preserve startup failure
                failures.append(cleanup)
        child=self._startup_child
        if child is not None:
            try:
                if self._startup_start_requested:
                    await child.close(deadline)
                else:
                    self._retire_unstarted_child(child)
            except BaseException as cleanup:  # noqa: BLE001 - retain every provisional owner and preserve startup failure
                failures.append(cleanup)
        if self.owned_registry is not None:
            try: self.owned_registry.close()
            except BaseException as cleanup:  # noqa: BLE001 - retain every provisional owner and preserve startup failure
                failures.append(cleanup)
        if self.online is not None: self.online.active=False
        state.update(status='failed',failure=type(error).__name__,quarantine=True)
        if failures:
            state['cleanup_failure']='; '.join(str(item) for item in failures)
            vars(error)['_hpo_owned_startup_owner']=self
            vars(error)['cleanup_owner']=self
            for cleanup_failure in failures: error.add_note('owned startup cleanup: '+str(cleanup_failure))
        try:
            self.persist_control(self.ownership.directory/'run.json',state,config,deadline=deadline)
        except BaseException as persistence:  # noqa: BLE001 - original startup error survives best-effort failed-state persistence
            error.add_note('owned startup failed-state persistence: '+str(persistence))
        if not failures:
            try: self.receipt=self.job.confirm_termination()
            except BaseException as cleanup:  # noqa: BLE001 - only positive actual ownership can issue termination
                vars(error)['_hpo_owned_startup_owner']=self
                error.add_note('owned startup termination proof: '+str(cleanup))

    def _retire_unstarted_child(self, child: Any) -> None:
        # Synchronous construction precedes our sole start dispatch. This
        # original lifecycle proof is separate from an absent PID/process.
        if child is not self._startup_child or self._startup_start_requested:
            raise ContractError('owned no-start retirement differs from original lifecycle')
        if (getattr(child,'creation_task',None) is not None or getattr(child,'lease',None) is not None
                or getattr(child,'process',None) is not None):
            raise ContractError('owned startup has unexpected process-creation custody')
        if hasattr(child,'channel') or hasattr(child,'_inherited'):
            child._require_original_unrequested_creator()
        for name in ('channel','_inherited'):
            actual=getattr(child,name,None)
            if actual is not None:
                if getattr(actual,'outstanding',()) or getattr(actual,'received_pending',()):
                    raise ContractError('unstarted owned channel has live transfer custody')
                actual.close()

    async def close_owned_child(self, child: Any, deadline: float) -> None:
        if not self._startup_start_requested: self._retire_unstarted_child(child)
        await child.close(deadline)

    async def start_owned(self, child: Any) -> None:
        if child is not self._startup_child or self._startup_start_requested:
            raise ContractError('owned startup requires original single start dispatch')
        self._startup_start_requested=True
        await child.start()

    async def initialize_owned(self, child: Any, seed: int, output: Path, *, deadline: float) -> Json:
        self.last_snapshot=await child.initialize_owned(seed,output,deadline=deadline)
        return {'hardware':child.hardware}

    @contextmanager
    def precollection_scope(self, plan: Any, ledger: Any, *, config: Any, deadline: float) -> Iterator[None]:
        yield

    def validate_owned_collection(self, plan: Any, ledger: Any, *, config: Any, deadline: float) -> None:
        from gf_training.hpo_owned_collection import _OwnedCollection
        registry=self.owned_registry
        assert registry is not None
        if registry.capture is None:
            registry.plan=plan
            registry.capture=_OwnedCollection(registry,plan,registry.binding(registry.current.generation)[0])
        pool=self.resident_runtime.pool
        if pool.admission is None or pool.admission.plan is not plan: pool.admit(plan)
        self.guard_collection(plan,ledger,resident_phase=pool.admission,deadline=deadline)

    def guard_collection(self, plan: Any, ledger: Any, *, resident_phase: Any, deadline: float) -> Any:
        from gf_training.hpo_owned_collection import _OwnedCollection
        from gf_training.hpo_resident import CollectionAdmission, ParkedAuthority
        registry=self.owned_registry
        if registry is None or self.online is None: raise ContractError('owned collection lacks live registry')
        self.online.check()
        with registry.parent_budget.reserve(1024**2+65536):
            registry.external_controls.finalize()
        if self.ledger is None or ledger is not self.ledger:
            raise ContractError('owned collection requires SAME actual learner ledger')
        if self.ledger.updating or time.monotonic()>=deadline:
            raise ContractError('owned collection requires SAME idle ledger and original deadline')
        if type(registry.capture) is not _OwnedCollection: raise ContractError('invalid owned collection attachment')
        registry.capture.require(registry,plan)
        if type(resident_phase) is CollectionAdmission:
            resident_phase.require(plan,ledger)
        elif type(resident_phase) is ParkedAuthority:
            resident_phase.require(ledger)
            if (resident_phase.evidence['generation']!=registry.current.generation
                    or resident_phase.evidence['collection_id']!=plan.linkage['batch_id']):
                raise ContractError('owned collection PARK differs from actual frozen plan')
        else: raise ContractError('owned collection requires actual collection/PARK phase')
        return self.online

    def binding(self, directory: Path, generation: int, config: Any, deadline: float) -> Any:
        if directory!=self.ownership.directory or time.monotonic()>=deadline: raise ContractError('owned binding outside original run/deadline')
        assert self.owned_registry is not None
        return self.owned_registry.binding(generation)

    def accept_collection(self, collection: Any, config: Any) -> None:
        from gf_training.hpo_owned_collection import OwnedFinalizedCollection
        if type(collection) is not OwnedFinalizedCollection or self.owned_registry is None:
            raise ContractError('actual capture and completed producer publication required')
        collection.require(self.owned_registry,self.owned_registry.plan)
        self.finalized_collection=collection

    @contextmanager
    def project_owned(self, collection: Any, plan: Any, numeric: Path, config: Any, deadline: float) -> Iterator[Any]:
        from gf_common.card_instance_rules import card_instance_rules
        registry=self.owned_registry
        assert registry is not None
        collection.require(registry,plan)
        selection=collection.selection()
        if selection['used_decisions']==0:
            yield 0,None,None,None
            return
        self.transaction=registry.encode_and_prepare(collection,numeric,
            model=config.model.architecture(layout=LAYOUT),training=hpo_training_config(config.hpo_spec['training_recipe']),
            actions=config.actions,rules=card_instance_rules(),numeric_budget_bytes=config.hpo_spec['operating']['numeric_budget_bytes'],
            authority=self.resident_runtime.pool.authority)
        self.numeric=self.transaction.numeric
        files=self.transaction.files
        primary: BaseException | None=None
        try:
            yield self.numeric.used_decisions,collection.proof(self.numeric),files,self.transaction
        except BaseException as error:
            primary=error
            raise
        finally:
            # Failure may leave dispatch-owned views until actual child reap.
            if self.numeric is not None and self.numeric._state!='DISPATCHED':
                try: self.close_batch()
                except BaseException as cleanup:
                    if primary is None: raise
                    vars(primary)['_hpo_batch_cleanup_owner']=self
                    primary.add_note('owned batch cleanup remains owned: '+str(cleanup))

    async def update_owned(self, transaction: Any, output: Path, *, deadline: float) -> Json:
        if transaction is not self.transaction: raise ContractError('wrong actual owned update transaction')
        self.last_snapshot=await transaction.dispatch(output,deadline=deadline)
        return {'metrics':self.last_snapshot.metrics,'timings':self.last_snapshot.timings}

    def update_record(self, update: Json, state: Json, numeric_files: Json) -> None:
        super().update_record(update,state,numeric_files)
        update.update(schema_version=6,origin='producer-owned-v1',encoder='canonical-python')

    def commit(self, directory: Path, state: Json, config: Any, *, generation: int, update: Json | None,
               controls: Json, acknowledgement: Json, deadline: float) -> Json:
        registry=self.owned_registry
        assert registry is not None
        if generation: assert update is not None
        result=(registry.commit_initial(self.last_snapshot,controls=controls,deadline=deadline) if generation==0 else
            registry.commit_update(self.transaction,update,controls=controls,deadline=deadline))  # type: ignore[arg-type]
        self.resident_runtime.pool.bind_owned_snapshot(registry.current.actor_capability)
        self.committed(state,config,None if update is None else update['metrics'])
        return result

    def retain(self, directory: Path, generation: int, config: Any, deadline: float) -> None:
        assert self.owned_registry is not None
        self.owned_registry.retain(config.hpo_spec['operating'],deadline=deadline,pool=self.resident_runtime.pool)

    def close_batch(self) -> None:
        if self.numeric is not None:
            self.numeric.close()
            self.numeric.require_retired()
            self.numeric=None
        if self.finalized_collection is not None:
            self.finalized_collection.batch.close()
            self.finalized_collection=None
        if self.owned_registry is not None:
            if self.owned_registry.capture is not None: self.owned_registry.capture.retire()
            if self.owned_registry._encoded is not None:
                self.owned_registry._encoded.close()
                self.owned_registry._encoded.require_retired()
                self.owned_registry._encoded=None
                self.owned_registry._encoded_collection=None
            self.owned_registry.capture=None
            self.owned_registry.plan=None
        self.transaction=None

    def finish_cleanup(self, error: BaseException | None) -> None:
        if self.owned_registry is not None: self.owned_registry.close()
        super().finish_cleanup(error)
