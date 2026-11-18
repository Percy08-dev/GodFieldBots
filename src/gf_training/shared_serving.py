"""Committed inference publication and service-shaped checkpoint reference custody."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, digest
from gf_common.ram_bank import ReadonlyRamBankSet
from gf_common.shared_inference import ServingBudget
from gf_models.sharded_arrays import read_bundle, read_json


@dataclass
class ServiceCacheAcquisition:
    service: Any
    binding: Json
    snapshot_binding: Any=None
    staging: Any=None


class SharedServingController:
    """No learning state or policy computation; retains actual service and run owners."""
    def __init__(self,operations: Any,config: Any,state: Json,service: Any,*,owned_registry: Any=None) -> None:
        self.operations,self.config,self.state,self.service=operations,config,state,service
        self.registry=owned_registry
        self.directory=operations.ownership.directory
        self.pins: dict[str,ServiceCacheAcquisition]={}
        self.budget=ServingBudget(service.runtime['cache_weight_bytes']*3+1024**2)
        self.publications: list[Json]=[]
        self.collection_usage: Json | None=None
        self._boundary={'requests':0,'publications':0}
        if state.get('invocations'):
            stamp=state['invocations'][-1]['shared_service']
            stamp['publications']=self.publications
            stamp.update(service_id=service.service_id,execution=service.execution,
                job_id=service.registration['job_id'],registration_sha256=digest(service.registration))

    def _registry(self) -> Json:
        if self.registry is not None:return self.registry.registry
        return read_json(self.directory/'artifact-registry.json')

    def _write(self,registry: Json) -> None:
        from gf_training.journal import atomic_json
        if self.registry is not None:
            self.registry.writer.write(self.directory/'artifact-registry.json',registry,deadline=self.service.deadline)
        else:atomic_json(self.directory/'artifact-registry.json',registry)

    def _record(self,token: str,phase: str,value: Json) -> None:
        folder=self.service.directory/'references'/token
        folder.mkdir(parents=True,exist_ok=True)
        self.service._persist('references/'+token+'/'+phase+'.json',value)

    def ensure(self,bindings: dict[str,Any],artifacts: Any) -> None:
        neural={binding.policy_options['expected_manifest_digest']:(label,binding)
            for label,binding in bindings.items() if binding.model_id=='mlp' and binding.policy_options is not None}
        runtime=self.service.runtime
        if len(neural)>runtime['cache_entries']:
            raise ContractError('complete frozen shared checkpoint inventory exceeds configured cache')
        for checkpoint in tuple(self.pins):
            if checkpoint not in neural:self.evict(checkpoint)
        for checkpoint,(label,binding) in neural.items():
            if checkpoint in self.pins:continue
            options=binding.policy_options
            path=Path(options['checkpoint'])
            if path.parent!=self.directory/'artifacts' or not path.name.isdecimal():
                raise ContractError('shared checkpoint escaped original committed parent')
            generation=int(path.name)
            # Validate metadata and construct the journal row before borrowing resources.
            manifest=artifacts.journal_metadata(label)['inference_manifest']
            registry=self._registry()
            snapshot=None
            if self.registry is not None:
                snapshot=self.registry.snapshots[generation].actor_capability
                snapshot.require()
                if snapshot.checkpoint!=checkpoint:raise ContractError('shared snapshot differs from committed digest')
            token=uuid.uuid4().hex
            row: Json={'schema_version':1,'kind':'shared-service-cache-reference','phase':'acquire','binding':{
                'run_id':self.state['run_id'],'token':token,'generation':generation,'inference_sha256':checkpoint,
                'generation_sha256':self.state['generation_sha256'] if generation==self.state['updates'] else self._generation_digest(generation),
                'policy_sha256':digest({'model_id':binding.model_id,'policy_options':options}),
                'weights_sha256':digest(manifest['weights']['files']),'snapshot_identity':None if snapshot is None else snapshot.identity,
                'service_id':self.service.service_id,'service_key':1,'broker_pid':self.service.registration['broker_pid'],
                'gpu_pid':self.service.registration['gpu_pid'],'cache_identity':digest({'checkpoint':checkpoint,'service_id':self.service.service_id})}}
            pending=ServiceCacheAcquisition(self.service,row['binding'])
            tracked=False
            try:
                if snapshot is not None:
                    pending.snapshot_binding=snapshot.bind(self)
                    banks=snapshot.inference
                else:
                    arrays=read_bundle(path,manifest['weights'],max_decoded_bytes=runtime['cache_weight_bytes'])
                    artifacts._scan.finalize()
                    pending.staging=ReadonlyRamBankSet.build(arrays,self.budget,max_bytes=runtime['cache_weight_bytes'])
                    del arrays
                    banks=pending.staging
                self.pins[checkpoint]=pending
                tracked=True  # All subsequent failures leave a reachable actual owner.
                registry.setdefault('service_references',{})[token]=row
                self._record(token,'acquire',row)
                self._write(registry)
                ack=self.service.publish(banks,model=options['model'],binding=options['inference'],checkpoint=checkpoint,
                    generation=generation,identity=row['binding']['snapshot_identity'] or checkpoint)
                from gf_common.domain import integer
                integer(ack.get('publication'),len(self.publications)+1,len(self.publications)+1)
                publication={'schema_version':1,'ordinal':ack['publication'],'token':token,'checkpoint':checkpoint,
                    'generation':generation,'ack_sha256':digest(ack)}
                self.publications.append(publication)
                self._record(token,'publication',publication|{'ack':ack})
                if pending.staging is not None:
                    pending.staging.close()
                    pending.staging=None
            finally:
                if not tracked:
                    if pending.snapshot_binding is not None:pending.snapshot_binding.close(self)
                    if pending.staging is not None:pending.staging.close()

    def _generation_digest(self,generation: int) -> str:
        import hashlib
        return hashlib.sha256((self.directory/'generations'/f'{generation:06}.json').read_bytes()).hexdigest()

    def _release(self,checkpoint: str,evidence: Json) -> None:
        acquisition=self.pins[checkpoint]
        if acquisition.service is not self.service:raise ContractError('foreign service cache acquisition')
        binding=acquisition.binding
        row={'schema_version':1,'kind':'shared-service-cache-reference','phase':'release','binding':binding,
            'evidence':evidence}
        try:
            self._record(binding['token'],'release',row)
            registry=self._registry()
            registry['service_references'][binding['token']]=row
            self._write(registry)
        finally:
            # The caller already holds actual eviction/reap proof; journal failure
            # invalidates the parent but cannot strand positively retired resources.
            if acquisition.snapshot_binding is not None:acquisition.snapshot_binding.close(self)
            if acquisition.staging is not None:acquisition.staging.close()
            del self.pins[checkpoint]

    def evict(self,checkpoint: str) -> None:
        ack=self.service.evict(checkpoint)
        self._release(checkpoint,{'kind':'shared-service-cache-eviction','service_id':self.service.service_id,
            'checkpoint':checkpoint,'ack':ack,'ack_sha256':digest(ack)})

    def after_commit(self,generation: int) -> None:
        # Source has retired and the generation chain is already durable.
        from gf_common.card_instance_rules import card_instance_rules
        from gf_models.card_instance_artifacts import RetainedGameArtifacts
        controls=(self.registry.records[-1]['parent_controls'] if self.registry is not None else
            read_json(self.directory/'generations'/f'{generation:06}.json')['parent_controls'])
        active=set(controls['snapshot_pool'])|{generation}
        bindings=dict(self.operations.binding(self.directory,g,self.config,self.service.deadline) for g in sorted(active))
        if self.registry is not None:
            artifacts=self.registry.artifacts(bindings)
        else:
            artifacts=RetainedGameArtifacts.for_registry(bindings,self.config.actions,rules=card_instance_rules(),
                max_owned_bytes=self.config.model.resources.max_owned_bytes,
                preflight_deadline=min(self.service.deadline,time.monotonic()+30.),finalization_deadline=self.service.deadline)
        try:self.ensure(bindings,artifacts)
        finally:artifacts.release()

    def finish_collection(self,plan: Any) -> Json:
        from gf_common.domain import clone
        self.service.quiesce()
        end={key:self.service.telemetry[key] for key in ('requests','publications')}
        required={label:{'checkpoint':binding.policy_options['expected_manifest_digest'],
            'token':self.pins[binding.policy_options['expected_manifest_digest']].binding['token']}
            for label,binding in plan.bindings.items() if binding.model_id=='mlp' and binding.policy_options is not None}
        self.collection_usage={'schema_version':1,'service_id':self.service.service_id,
            'execution':self.service.execution,'job_id':self.service.registration['job_id'],
            'registration_sha256':digest(self.service.registration),'config_sha256':self.state['config_sha256'],
            'runtime_sha256':digest(self.service.runtime),'generation':plan.linkage['collection_generation'],
            'batch_id':plan.linkage['batch_id'],'start':dict(self._boundary),'end':end,'bindings':required}
        self._boundary=end
        return clone(self.collection_usage)

    def close(self) -> Json:
        failure: BaseException | None=None
        try:proof=self.service.close()
        except BaseException as error:  # Actual proof can survive a later persistence/close failure.
            if not self.service.closed or self.service.proof is None:raise
            self.service.require_retirement(self.service.proof)
            proof=self.service.proof
            failure=error
        for checkpoint in tuple(self.pins):
            try:self._release(checkpoint,{'kind':'shared-service-cache-reaped','service_id':self.service.service_id,'resource':proof})
            except BaseException as error:  # noqa: BLE001 - every independently retired pin must release.
                if failure is None:failure=error
                else:failure.add_note('additional service pin cleanup: '+str(error))
        if failure is not None:raise failure
        if self.service.failed:
            raise RuntimeError('shared inference invocation failed after confirmed operational retirement')
        return proof


def validate_service_references(registry: Json,state: Json,records: list[Json],scan: Any) -> None:
    """Representation/original-file audit only; persisted JSON grants no custody."""
    import re

    from gf_common.domain import closed, integer
    from gf_models.structured_sealed import checksum
    from gf_training.shared_inventory import validate_inventory
    services=validate_inventory(registry,state,records,scan)
    refs=registry.get('service_references')
    if type(refs) is not dict:raise ContractError('shared registry requires explicit service references')
    for token,row in refs.items():
        closed(row,{'schema_version','kind','phase','binding'}|({'evidence'} if row.get('phase')=='release' else set()))
        binding=closed(row['binding'],{'run_id','token','generation','inference_sha256','generation_sha256',
            'policy_sha256','weights_sha256','snapshot_identity','service_id','service_key','broker_pid','gpu_pid','cache_identity'})
        generation=integer(binding['generation'],0,len(records)-1)
        integer(row['schema_version'],1,1)
        integer(binding['service_key'],1,1)
        if (row['schema_version']!=1 or row['kind']!='shared-service-cache-reference' or binding['run_id']!=state['run_id']
                or binding['token']!=token or binding['inference_sha256']!=records[generation]['inference_sha256']):
            raise ContractError('shared service pin differs from committed original')
        for key in ('token','service_id'):
            if type(binding[key]) is not str or re.fullmatch('[0-9a-f]{32}',binding[key]) is None:
                raise ContractError('shared service reference identity required')
        for key in ('inference_sha256','generation_sha256','policy_sha256','weights_sha256','cache_identity'):
            checksum(binding[key])
        if (binding['generation_sha256']!=scan.verified_digest(scan.root/'generations'/f'{generation:06}.json')
                or binding['weights_sha256']!=digest({name:sha for name,sha in records[generation]['artifact_files'].items() if name.startswith('weights-')})
                or binding['snapshot_identity']!=records[generation].get('snapshot_identity')
                or binding['cache_identity']!=digest({'checkpoint':binding['inference_sha256'],'service_id':binding['service_id']})):
            raise ContractError('shared cache pin immutable generation/weights identity differs')
        registration=services[binding['service_id']]['registration']
        if registration['broker_pid']!=binding['broker_pid'] or registration['gpu_pid']!=binding['gpu_pid']:
            raise ContractError('shared cache process differs from original invocation')
        raw_model={key:value for key,value in state['effective_config']['model'].items()
            if key not in ('checkpoint','inference','cuda_visible_devices')}
        inference=state['execution']['inference']
        options={'version':state['model_version'],'checkpoint':str(scan.root/'artifacts'/f'{generation:06}'),
            'expected_manifest_digest':binding['inference_sha256'],'model':raw_model,'inference':inference['binding'],
            'cuda_visible_devices':inference['cuda_visible_devices'],'action_config_hash':digest(state['effective_config']['actions']),
            **{key:state[key] for key in ('action_contract','rule_signature','layout_signature','purpose','synthetic_capacity')}}
        if binding['policy_sha256']!=digest({'model_id':state['model_id'],'policy_options':options}):
            raise ContractError('shared service pin policy differs from committed execution')
        folder=scan.root/'services'/binding['service_id']/'references'/token
        acquired=scan.object(folder/'acquire.json')
        if acquired!={key:value for key,value in row.items() if key!='evidence'} | {'phase':'acquire'}:
            raise ContractError('shared acquisition original differs')
        if row['phase']=='release':
            if scan.object(folder/'release.json')!=row:raise ContractError('shared service release original differs')
            evidence=row['evidence']
            if evidence['service_id']!=binding['service_id']:raise ContractError('foreign shared pin release')
            if evidence['kind']=='shared-service-cache-eviction':
                closed(evidence,{'kind','service_id','checkpoint','ack','ack_sha256'})
                if (evidence['checkpoint']!=binding['inference_sha256'] or evidence['ack']!={'accepted':True,'checkpoint':binding['inference_sha256']}
                        or evidence['ack_sha256']!=digest(evidence['ack'])):
                    raise ContractError('shared cache release lacks original eviction acknowledgement')
            elif evidence['kind']=='shared-service-cache-reaped':
                closed(evidence,{'kind','service_id','resource'})
                proof=scan.object(scan.root/'services'/binding['service_id']/'proof.json')
                if evidence['resource']!=proof:raise ContractError('shared cache release composite reap differs')
            else:raise ContractError('unknown shared cache release kind')
        elif row['phase']!='acquire' or state['status']!='running':
            raise ContractError('shared cold parent retains unresolved service cache custody')
