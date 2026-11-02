"""Actor-only final NN validation ownership; collection evidence, not qualification."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Self

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, Json, RuntimePolicyBinding, canonical, digest
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.structured import ActionConfigV2
from gf_models.config import PPOTrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV2, recorded_policy_label
from gf_models.structured_config import InferenceBindingV2, ModelConfigV2
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.collection import CollectionPlan, validation_groups
from gf_training.lineage import _OWNED_LOCKS
from gf_training.native_structured_session import NativeStructuredSource
from gf_training.structured_rollout import _owned_bytes
from gf_training.structured_tree_data import _check_output_ancestors, _outside_output


def _checked_returned_report(persisted:Json,returned:Json,elapsed:Any,wall:Any)->None:
    """Only the CLI's documented post-write wall measurement may advance."""
    if set(persisted)!=set(returned) or 'wall_seconds' not in persisted:
        raise ContractError('final validation returned report fields differ from original file')
    if canonical({key:value for key,value in persisted.items() if key!='wall_seconds'})!=canonical(
            {key:value for key,value in returned.items() if key!='wall_seconds'}):
        raise ContractError('final validation returned report content differs from original file')
    times=(persisted['wall_seconds'],elapsed,returned['wall_seconds'],wall)
    try:
        valid=all(type(value) in (int,float) and math.isfinite(value) and value>=0 for value in times)
    except OverflowError:
        valid=False
    if not valid or not 0<=times[0]<=times[1]<=times[2]<=times[3]:
        raise ContractError('final validation persisted/manifest/returned wall timing order invalid')


class FinalValidationContext:
    """One completed parent, final policy and SAME ledger with no learner.

    The private component collector supplies fixed I/O doubles. This context is
    not permission to bypass the independently guarded normal Environment.
    """
    def __init__(self,loaded:Any,parent:Path,output:Path,ledger:ReservationLedger,*,source_settings:PreparationConfig,deadline:float,native:Any=None)->None:
        cfg=loaded.config
        if (not isinstance(cfg.actions,ActionConfigV2) or cfg.actions.action_contract!=engineering_contract()
                or not isinstance(cfg.model,ConfiguredNeuralModelV2) or cfg.model.checkpoint is not None
                or not isinstance(cfg.training,PPOTrainingConfig) or cfg.run.split!='train' or cfg.server.external_url is not None):
            raise ContractError('known engineering current NN parent Config required for component validation')
        if type(ledger) is not ReservationLedger:raise ContractError('actual actor-only ledger required')
        expected_source=preparation_config(asdict(replace(cfg.preprocessing,backend='rust',reader='structured-v1')))
        self.source_settings=preparation_config(asdict(source_settings))
        if canonical(asdict(self.source_settings))!=canonical(asdict(expected_source)):
            raise ContractError('actor source_settings may change ONLY backend/reader of recorded preparation')
        self.loaded=loaded;self.parent=Path(parent).absolute();self.ledger=ledger
        self.output,self._output_ancestors=_outside_output(Path(output),[self.parent])
        self.parent=self.parent.resolve(strict=False)
        self.started=time.monotonic();self.deadline=min(deadline,self.started+cfg.run.wall_seconds)
        self.native=native;self._source:NativeStructuredSource|None=None
        self.projected:Any=None;self.plan:CollectionPlan|None=None;self._expected:str|None=None
        self._closed=False;self._collected=False;self._lock_token:str|None=None

    def _extra(self)->int:
        # Rollout ownership deliberately accepts only its closed raw types.
        # Count the control plan's owned JSON representation, rather than
        # passing its filesystem Path into that model-neutral estimator.
        plan_bytes=0 if self.plan is None else 32*len(self._plan_signature(self.plan).encode())+65536
        if self.plan is not None:
            plan_bytes+=32*len(canonical({**asdict(self.plan),'output':str(self.plan.output)}).encode())
        control={'parent':str(self.parent),'output':str(self.output),'source_settings':asdict(self.source_settings),
                 'output_ancestors':{str(path):identity for path,identity in self._output_ancestors.items()}}
        return _owned_bytes((self.loaded,self.projected,self._expected,control))+plan_bytes

    def _actor_ledger(self)->None:
        cfg=self.loaded.config
        if (self.ledger.learner is not None or self.ledger.learner_bytes or self.ledger.updating or self.ledger.workers
                or self.ledger.global_cuda_bytes!=cfg.run.global_cuda_budget_bytes
                or self.ledger.max_workers!=cfg.run.max_participant_workers):
            raise ContractError('final validation requires SAME empty actor-only ledger and frozen limits')

    async def __aenter__(self)->Self:
        try:
            self._actor_ledger()
            cfg=self.loaded.config
            assert isinstance(cfg.actions,ActionConfigV2) and isinstance(cfg.model,ConfiguredNeuralModelV2) and isinstance(cfg.training,PPOTrainingConfig)
            plan={'schema_version':2,'mode':'historical_parent','parents':[{'directory':str(self.parent),'require_complete':True}],'collections':[]}
            self._source=NativeStructuredSource(plan,self.source_settings,max_owned_bytes=min(cfg.training.max_owned_bytes,cfg.model.resources.max_owned_bytes),
                deadline=self.deadline,extra_owned_bytes=self._extra(),native=self.native)
            self._source.__enter__();assert self._source.summary is not None
            parent=self._source.summary['parents'][0]
            if (parent['state']['config_sha256']!=digest(asdict(cfg)) or type(self._source.models[0]) is not ModelConfigV2
                    or canonical(parent['model'])!=canonical(asdict(cfg.model.architecture()))):
                raise ContractError('final validation frozen caller/current376 parent mismatch')
            original_inputs=self._source.original_config_inputs(0,self._extra())
            if canonical(original_inputs)!=canonical(list(self.loaded.inputs)):
                raise ContractError('final validation caller input origins differ from verified original parent')
            self._lock_token=_OWNED_LOCKS.get(self.parent.resolve())
            if self._lock_token is None:raise ContractError('actual completed-parent lock required')
            model=self._source.models[0];record=parent['generations'][-1]
            binding=InferenceBindingV2(**parent['execution']['inference']['binding'])
            visibility=parent['execution']['inference']['cuda_visible_devices']
            label=recorded_policy_label(model,binding,record['inference_sha256'],cuda_visible_devices=visibility)
            options={'version':'2','checkpoint':str(self.parent/'artifacts'/f"{record['generation']:06}"),
                'expected_manifest_digest':record['inference_sha256'],'model':asdict(model),'inference':asdict(binding),
                'cuda_visible_devices':visibility,'action_contract':asdict(cfg.actions.action_contract),'purpose':'component','synthetic_capacity':False}
            registry={label:RuntimePolicyBinding(model.model_id,options),'random-structured-v1':RuntimePolicyBinding('random'),
                'heuristic-structured-v1':RuntimePolicyBinding('heuristic')}
            projected=replace(cfg,run=replace(cfg.run,split='validation'))
            self.projected=replace(self.loaded,config=projected,effective=asdict(projected),sha256=digest(asdict(projected)))
            groups=validation_groups(parent['state']['seed'],label,actions_version=2)
            seats={spec.episode_id:spec.learner_viewer for group in groups for spec in group.episodes}
            self.plan=CollectionPlan(groups,registry,self.output,{'parent_run_id':parent['state']['run_id'],
                'parent_run_path':str(self.parent),'batch_id':'validation-final','collection_generation':record['generation'],'learner_seats':seats},
                min(self.deadline,self.started+cfg.run.execution_seconds),self.deadline,'ppo_validation',self.started)
            self._expected=self._plan_signature(self.plan)
            self.validate_collection(self.plan,self.ledger,self.projected.config,deadline=self.deadline)
            return self
        except BaseException as error:
            try:await self.aclose()
            except BaseException as cleanup:  # noqa: BLE001 - retain the actual unconfirmed owner
                error.add_note('final validation context cleanup: '+type(cleanup).__name__)
                owner=getattr(cleanup,'_structured_source_cleanup_owner',None)
                if owner is not None:vars(error)['_structured_source_cleanup_owner']=owner
            raise

    @staticmethod
    def _plan_signature(plan:CollectionPlan)->str:
        return digest({'groups':asdict(plan)['groups'],'bindings':{k:asdict(v) for k,v in plan.bindings.items()},
            'output':str(plan.output),'linkage':plan.linkage,'kind':plan.kind,
            'execution_deadline':plan.execution_deadline,'wall_deadline':plan.wall_deadline,'started_monotonic':plan.started_monotonic})

    def validate_collection(self,plan:CollectionPlan,ledger:ReservationLedger,config:Any,*,deadline:float)->None:
        if self._closed or self._source is None or self.plan is None or self.projected is None:
            raise ContractError('live actor-only final validation context required')
        if time.monotonic()>=min(deadline,self.deadline):raise TimeoutError('final validation context deadline')
        if (plan is not self.plan or ledger is not self.ledger or self._plan_signature(plan)!=self._expected
                or canonical(asdict(config))!=canonical(asdict(self.projected.config))
                or _OWNED_LOCKS.get(self.parent.resolve())!=self._lock_token):
            raise ContractError('final validation context/frozen plan/Config/ledger ownership mismatch')
        _check_output_ancestors(self._output_ancestors)
        self._actor_ledger();self._source.recheck(self._extra())

    def accept_collection(self,directory:Path,report:Json)->Json:
        assert self.plan is not None and self.projected is not None and self._source is not None
        self.validate_collection(self.plan,self.ledger,self.projected.config,deadline=self.deadline)
        if Path(directory).absolute()!=self.output or self._collected:
            raise ContractError('one exact final validation collection required')
        if (report.get('planned_games')!=64 or report.get('natural_denominator')!=64 or report.get('comparison_valid') is not True
                or any(report.get(name) for name in ('failures','health_failures','group_health_failures'))):
            raise ContractError('all64 healthy natural final validation games required')
        from gf_common.schedule import serialize_schedule
        from gf_models.structured_sealed import SealedScan
        remaining=self._source.budget-self._source._session.used_bytes()-self._extra()
        if remaining<=0:raise ContractError('joint final validation metadata reservation exhausted')
        scan=SealedScan(max_owned_bytes=remaining,deadline=self._source.deadline)
        manifest=scan.object(self.output/'manifest.json')
        if (manifest.get('schema_version')!=4 or manifest.get('status')!='complete' or manifest.get('collection_kind')!='ppo_validation'
                or manifest.get('split')!='validation' or manifest.get('config_sha256')!=self.projected.sha256
                or canonical(manifest.get('effective_config'))!=canonical(asdict(self.projected.config))
                or canonical(manifest.get('config_inputs'))!=canonical(list(self.loaded.inputs))
                or canonical(manifest.get('ppo'))!=canonical(self.plan.linkage)
                or canonical(manifest.get('schedule'))!=canonical(serialize_schedule(self.plan.groups,4))
                or canonical(manifest.get('policy_bindings'))!=canonical({key:asdict(value) for key,value in self.plan.bindings.items()})):
            raise ContractError('final validation original manifest differs from exact projected plan')
        observed=scan.object(self.output/'report.json',manifest['report_sha256'])
        _checked_returned_report(observed,report,manifest['elapsed_seconds'],self.loaded.config.run.wall_seconds)
        scan.finalize();self._source.recheck(self._extra()+scan.retained)
        self._collected=True
        return {'directory':str(self.output),'manifest_sha256':scan.files[self.output/'manifest.json'][1],
            'parent_run_id':self.plan.linkage['parent_run_id'],'final_generation':self.plan.linkage['collection_generation'],
            'original_config_sha256':digest(asdict(self.loaded.config)),'validation_config_sha256':self.projected.sha256,
            'qualified':False,'parent_qualification':'not_performed','source_inspection':{'binary':self._source.identity,'settings':asdict(self.source_settings)},'scope':'collection-metadata-only; DatasetSession audits raw validation Source'}

    async def aclose(self)->None:
        if self._closed:return
        failure:BaseException|None=None
        if self.ledger.workers:
            failure=failure or ContractError('final validation still owns unreaped actor processes')
            vars(failure)['_structured_source_cleanup_owner']=self;raise failure
        if self._source is not None:
            try:self._source.close()
            except BaseException as error:  # noqa: BLE001 - NativeExecution may preserve cancellation after confirmed join
                failure=failure or error
                if not self._source._closed:
                    vars(failure)['_structured_source_cleanup_owner']=getattr(error,'_structured_source_cleanup_owner',self._source)
                    raise failure
        self._closed=True
        if failure is not None:raise failure

    async def __aexit__(self,kind:object,value:object,traceback:object)->None:
        failure=None
        if value is None:
            try:
                if not self._collected:raise ContractError('actor validation context has no completed collection evidence')
                assert self._source is not None
                self._source.finalize()
            except BaseException as error:  # noqa: BLE001 - finalize before confirmed owned cleanup
                failure=error
        try:await self.aclose()
        except BaseException as cleanup:
            original=value if isinstance(value,BaseException) else failure
            if original is None:raise
            original.add_note('final validation cleanup: '+type(cleanup).__name__)
            owner=getattr(cleanup,'_structured_source_cleanup_owner',None)
            if owner is not None:vars(original)['_structured_source_cleanup_owner']=owner
        if failure is not None:raise failure
