"""Concrete HPO operations for the one shared parent coordinator."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_binding import ConfiguredNeuralModelV4, input_binding_game_v3
from gf_models.structured_config import InferenceBindingV2
from gf_training.card_instance_ppo import _GameOperations
from gf_training.card_instance_source import _OnlineGameOwner, frozen_linkage
from gf_training.hpo_artifacts import checked_phase_limits, save_estimate
from gf_training.hpo_collection import training_groups_hpo
from gf_training.hpo_config import (
    effective_settings,
    hpo_training_config,
    normalize_training_for_model,
)
from gf_training.hpo_gpu_lease import GpuJob, TerminationReceipt, require_training_job
from gf_training.hpo_learning_process import HpoLearningProcess
from gf_training.hpo_lineage import (
    _run_projection,
    _runtime_training,
    commit_hpo_generation,
    hpo_spec,
    validate_hpo_chain,
    write_hpo_control,
)
from gf_training.hpo_retention import retain_hpo_generations
from gf_training.hpo_source import hpo_batch_source, hpo_binding, validate_hpo_parent
from gf_training.structured_lineage import _ChainScan


@dataclass(frozen=True)
class HpoRunResult:
    report: Json
    gpu_termination: TerminationReceipt


class _HpoOperations(_GameOperations):
    registry_schema=4

    def __init__(self,job: GpuJob):
        if type(job) is not GpuJob: raise ContractError('actual caller GpuJob required')
        super().__init__()
        self.job=job
        self.receipt: TerminationReceipt | None=None
        self.stop_reason: str | None=None

    def validate_config(self,config: Any) -> Any:
        if type(config.actions) is not ActionConfigV3 or type(config.model) is not ConfiguredNeuralModelV4 or config.model.checkpoint is not None:
            raise ContractError('HPO requires exact fresh Model4/actions3')
        raw=asdict(config)
        spec=hpo_spec({'effective_config':raw,'hpo_spec_sha256':digest(config.hpo_spec)})
        recipe=hpo_training_config(spec['training_recipe'])
        training=_runtime_training(raw['training'],spec)
        _run_projection(raw,spec,training)
        if canonical(asdict(normalize_training_for_model(config.model.model_id,recipe)))!=canonical(spec['training_recipe']):
            raise ContractError('HPO production recipe must already be canonical')
        from gf_common.training_opponents import training_opponents
        if training_opponents(raw) is not None:
            from gf_training.native_backend import require_native_refresh_opponents
            require_native_refresh_opponents()
        op=spec['operating']
        require_training_job(self.job,ledger=self.job.ledger,run_id=config.run.name,
            global_cuda_bytes=op['managed_cuda_bytes'],max_workers=op['max_participant_workers'])
        model=config.model.architecture(layout=LAYOUT)
        if effective_settings(model,recipe,{},layout=LAYOUT).identity!=spec['effective_settings_sha256']:
            raise ContractError('HPO declared effective settings mismatch before learner allocation')
        if save_estimate(model)['additional_bytes']>op['save_budget_bytes']:
            raise ContractError('HPO additional save envelope too small before allocation')
        if config.preprocessing.backend=='rust':
            from gf_training.native_backend import hpo_reader_settings, load_native
            self.native=load_native()
            hpo_reader_settings(config.preprocessing,self.native,packed=hasattr(config,'storage'))
        return model

    def preflight(self,config: Any,*,deadline: float) -> None:
        self.validate_config(config)
        if config.training.device!='cuda:0' or config.model.inference.device!='cuda:0' or not hasattr(config,'storage'):
            raise ContractError('HPO production requires actual CUDA training/inference and compressed original storage')
        from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
        if config.server.external_url is not None or config.server.source_dir is None:
            raise ContractError('HPO requires locally managed verified runtime')
        self.runtime=CardInstanceRuntimeEvidence(Path(config.server.binary),Path(config.server.source_dir),deadline=deadline)
        self.runtime.preflight()
        from gf_models.structured_runtime import admit_hpo_training
        admit_hpo_training(config.actions,config.model.architecture(layout=LAYOUT),config.model.inference,
            config.hpo_spec['training_recipe']['tbptt_length'],config.hpo_spec['operating'],rules=card_instance_rules())

    def initialize_state(self,state: Json,config: Any) -> None:
        from gf_training.structured_ppo import execution
        classification=input_binding_game_v3(config.actions.action_contract,card_instance_rules())
        state.update(schema_version=5,run_variant='hpo-ppo-v1',model_id=config.model.model_id,model_version='4',
            input_recipe=LAYOUT.input_recipe,execution=execution(config),hpo_spec_sha256=digest(config.hpo_spec),
            used_decisions=0,collected_decisions=0,stop_reason=None,**classification)
        if hasattr(config,'storage'):
            from gf_common.group_pack_game import storage_binding
            state['storage_binding']=storage_binding()
        state['resume_identity']=digest({'version':'hpo-ppo-resume-v1','config_sha256':state['config_sha256'],
            'identity':state['identity'],'execution':state['execution'],'classification':classification,
            'storage_binding':state.get('storage_binding')})

    def cap(self,config: Any) -> int:
        return min(config.training.max_owned_bytes,config.hpo_spec['operating']['source_budget_bytes'],config.preprocessing.memory_budget_mib*1024**2)

    def controls(self,rng: Any,pool: list[int],position: int) -> Json:
        return {'schedule_opponent_rng':rng.bit_generator.state,'snapshot_pool':pool,'collection_position':position}

    def persist_control(self,path: Path,value: Json,config: Any,*,deadline: float,original: Any=None) -> None:
        scan=original or _ChainScan(directory=path.parent,max_owned_bytes=min(self.cap(config),config.hpo_spec['operating']['parent_budget_bytes']),deadline=deadline)
        try:
            if path.exists() and path not in scan.files: scan.object(path)
            write_hpo_control(path,value,scan=scan,ownership=self.ownership)
            scan.finalize()
        finally:
            if original is None: scan.close(recheck=False)

    def read_control(self,path: Path,config: Any,*,deadline: float) -> Any:
        if path.name=='manifest.json': return super().read_control(path,config,deadline=deadline)
        scan=_ChainScan(directory=path.parent,max_owned_bytes=min(self.cap(config),config.hpo_spec['operating']['parent_budget_bytes']),deadline=deadline)
        try:
            value=scan.object(path)
            scan.finalize()
            return value,scan
        except BaseException:
            scan.close(recheck=False)
            raise

    def make_child(self,directory: Path,config: Any,state: Json,*,deadline: float) -> Any:
        op=config.hpo_spec['operating']
        ledger=self.job.ledger
        require_training_job(self.job,ledger=ledger,run_id=state['run_id'],global_cuda_bytes=op['managed_cuda_bytes'],max_workers=op['max_participant_workers'])
        child=HpoLearningProcess(directory,config.training.max_log_bytes,config.model.architecture(layout=LAYOUT),
            hpo_training_config(config.hpo_spec['training_recipe']),InferenceBindingV2(**state['execution']['learner']['binding']),
            digest(asdict(config.actions)),state['resume_identity'],actions=config.actions,ledger=ledger,
            cuda_visible_devices=state['execution']['learner']['cuda_visible_devices'],cleanup_deadline=deadline,
            storage_binding=state.get('storage_binding'),phase_limits=checked_phase_limits({key:op[key] for key in ('numeric_budget_bytes','save_budget_bytes')}))
        self.online=_OnlineGameOwner(self.ownership,state,child,ledger,hpo_job=self.job)
        self.ledger=ledger
        child.online_owner=self.online
        return child,ledger

    def validate_parent(self,directory: Path,config: Any,deadline: float,*,extra_owned_bytes: int=0) -> Any:
        return validate_hpo_parent(directory,require_complete=False,max_owned_bytes=self.cap(config),deadline=deadline,extra_owned_bytes=extra_owned_bytes)

    def binding(self,directory: Path,generation: int,config: Any,deadline: float) -> Any:
        state,records=validate_hpo_chain(directory,max_owned_bytes=self.cap(config),deadline=deadline)
        return hpo_binding(directory,state,records[generation],generation)

    def groups(self,config: Any,generation: int,label: str,snapshots: tuple[str,...],rng: Any) -> Any:
        from gf_common.training_opponents import training_opponents
        mixed = training_opponents(asdict(config))
        if mixed is not None:
            from gf_training.hpo_collection import training_groups_hpo_refresh
            return training_groups_hpo_refresh(config.training.seeds[0],generation,label,snapshots,rng,
                collection_games=config.hpo_spec['training_recipe']['collection_games'],
                actions=config.actions,training_opponents=mixed)
        return training_groups_hpo(config.training.seeds[0],generation,label,snapshots,rng,
            collection_games=config.hpo_spec['training_recipe']['collection_games'])

    def linkage(self,state: Json,directory: Path,generation: int,groups: Any,registry: Any) -> Json:
        from gf_common.hpo import selection_context
        link=frozen_linkage(state,directory,generation,groups,registry)
        link['selection_context']=selection_context(decision_budget=hpo_spec(state)['decision_budget'],
            used_before=state['used_decisions'],learning_seed=state['seed'],batch=generation)
        return link

    @contextmanager
    def project(self,plan: Any,numeric: Path,config: Any,learner_label: str,generation: int,deadline: float,*,
                expected_manifest_sha256: str,extra_owned_bytes: int=0) -> Iterator[Any]:
        from gf_models.arrays import sha256
        with hpo_batch_source(plan,self.ledger,max_owned_bytes=self.cap(config),deadline=deadline,extra_owned_bytes=extra_owned_bytes) as source:
            if source.proof['manifest_sha256']!=expected_manifest_sha256 or source.learner_label!=learner_label:
                raise ContractError('HPO original Source differs from callback result')
            proof=source.proof
            count=0
            files=None
            handle=None
            if source.selection['used_decisions']:
                manifest=source.write_numeric(numeric)
                files={'rollout.json':sha256(numeric/'rollout.json',deadline),**manifest['data']['files']}
                rollout=source.read_numeric(numeric,files['rollout.json'])
                rollout.require_learning()
                count=len(rollout)
                del rollout
                handle=source.issue(numeric,files)
            self.projection=handle
            primary: BaseException | None=None
            try:
                yield count,proof,files,handle
            except BaseException as error:
                primary=error
                raise
            finally:
                self.projection=None
                # Capture the actual reachable owner before close can change
                # its slot. A secondary close failure cannot replace the update.
                owner=self.online.packed_batch if self.online is not None else None
                if owner is None: owner=self
                try:
                    self.close_batch()
                except BaseException as cleanup:
                    actual=getattr(cleanup,'cleanup_owner',None)
                    if actual is None: actual=owner
                    failed=cleanup if primary is None else primary
                    if getattr(failed,'cleanup_owner',None) is None:
                        failed.__dict__['cleanup_owner']=actual
                    failed.__dict__['_hpo_batch_cleanup_owner']=actual
                    if primary is None:
                        raise
                    primary.add_note('packed batch cleanup: '+str(cleanup))

    def update_record(self,update: Json,state: Json,numeric_files: Json) -> None:
        proof=update['source_proof']
        selection=proof['selection']
        update.update(schema_version=5,model_id=state['model_id'],model_version='4',input_recipe=state['input_recipe'],
            **{key:state[key] for key in ('action_contract','purpose','synthetic_capacity','rule_signature','layout_signature')},
            numeric_files=numeric_files,hpo_spec_sha256=state['hpo_spec_sha256'],
            collected_decisions=selection['collected_decisions'],used_decisions=selection['used_decisions'],
            overshoot_decisions=selection['collected_decisions']-selection['used_decisions'],
            expected_optimizer_step_delta=proof['expected_optimizer_step_delta'])
        if 'storage_binding' in state: update['storage_binding']=state['storage_binding']

    def commit(self,directory: Path,state: Json,config: Any,*,generation: int,update: Json | None,
               controls: Json,acknowledgement: Json,deadline: float) -> Json:
        return commit_hpo_generation(directory,state,generation=generation,update=update,controls=controls,
            artifact_files=acknowledgement['artifact']['files'],max_owned_bytes=self.cap(config),deadline=deadline,
            projection=self.projection if generation else None,ownership=self.ownership)

    def retain(self,directory: Path,generation: int,config: Any,deadline: float) -> None:
        retain_hpo_generations(directory,generation,max_owned_bytes=self.cap(config),deadline=deadline,ownership=self.ownership)

    def stop(self,state: Json,config: Any,usage: int,now: float,execution: float,admission: float) -> bool:
        if state['used_decisions']==config.hpo_spec['decision_budget']: return True
        if usage>config.training.max_owned_bytes or now>=execution:
            raise RuntimeError('HPO actual execution/storage limit exceeded')
        if state['updates']>=config.training.max_updates: self.stop_reason='update_cap'
        elif now>=admission: self.stop_reason='time_cap'
        elif usage+config.training.admission_reservation_bytes>config.training.max_owned_bytes: self.stop_reason='storage_cap'
        return self.stop_reason is not None

    def finish_cleanup(self,error: BaseException | None) -> None:
        if error is not None and any(getattr(error,name,None) is not None for name in
                ('cleanup_owner','_game_artifact_cleanup_owner','_lineage_publication_owner','_hpo_retention_pending','_hpo_storage_uncertain')):
            raise ContractError('HPO resource/storage cleanup remains uncertain')
        if self.online is not None:
            if self.online.issued is not None or self.online.packed_batch is not None:
                raise ContractError('HPO Source/custody remains live')
            if self.online.collection_controls is not None: self.online.collection_controls.finalize()
            self.online.active=False
        self.receipt=self.job.confirm_termination()


async def train_hpo_ppo(loaded_run: Any,collector: Any,directory: Path,*,gpu_job: GpuJob,
                         mode: str='fresh',pause_after_update: int | None=None) -> HpoRunResult:
    operations=_HpoOperations(gpu_job)
    if canonical(asdict(loaded_run.config))!=canonical(loaded_run.effective) or digest(loaded_run.effective)!=loaded_run.sha256:
        raise ContractError('HPO loaded runtime config is not its exact original effective projection')
    directory=directory.resolve()
    declared=Path(loaded_run.config.hpo_spec['storage']['parent_root']).resolve()
    if (directory==declared or not directory.is_relative_to(declared)
            or loaded_run.config.run.output_dir!=str(declared)):
        raise ContractError('HPO parent must be a run directory under its declared colocated root')
    operations.validate_config(loaded_run.config)
    from gf_training.ppo_run import _run_ppo
    try:
        report=await _run_ppo(loaded_run,collector,directory,mode=mode,pause_after_update=pause_after_update,_game_ops=operations)
    except BaseException as error:
        if operations.receipt is not None: error.__dict__['hpo_gpu_termination']=operations.receipt
        raise
    if operations.receipt is None: raise ContractError('HPO run has no actual GPU termination receipt')
    return HpoRunResult(report,operations.receipt)
