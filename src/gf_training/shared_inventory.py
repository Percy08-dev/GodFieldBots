"""Small shared-only committed controls; no policy data or live authority."""
from __future__ import annotations

import re
from typing import Any, cast

from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.shared_inference import checked_usage
from gf_models.structured_sealed import checksum
from gf_training.hpo_update_storage import read_hpo_update

STAMP_FIELDS={'schema_version','finished','start_generation','end_generation','service_id','execution',
    'job_id','registration_sha256','proof_sha256','termination_sha256','publications','tail_usage'}



def new_invocation(generation: int) -> Json:
    return {'schema_version':1,'finished':False,'start_generation':generation,'end_generation':None,
        'service_id':None,'execution':None,'job_id':None,'registration_sha256':None,'proof_sha256':None,
        'termination_sha256':None,'publications':[],'tail_usage':None}


def finish_invocation(operations: Any,state: Json,stamp: Json,*,error: BaseException | None=None) -> None:
    controller=getattr(operations,'shared_controller',None)
    stamp.update(finished=True,end_generation=state['updates'])
    service=getattr(controller,'service',None)
    if service is None:
        lease=getattr(getattr(operations,'ledger',None),'service',None)
        service=getattr(lease,'owner',None) or getattr(error,'_shared_service_owner',None)
    if service is None:return
    if service.process is None and service.closed and getattr(service,'_no_child_error',None) is not None:
        service.require_retirement(service.proof)  # Original positive no-child observation, not an absent PID guess.
        return
    registration=service.registration
    stamp.update(service_id=service.service_id,execution=service.execution,
        job_id=(registration['job_id'] if registration is not None else getattr(getattr(operations,'job',None),'ticket',service.execution)),
        registration_sha256=None if registration is None else digest(registration),
        publications=[] if controller is None else clone(controller.publications),
        proof_sha256=None if service.proof is None else digest(service.proof))
    usage=getattr(controller,'collection_usage',None)
    if usage is not None and usage['generation']==state['updates']:stamp['tail_usage']=clone(usage)
    if state.get('run_variant')=='hpo-ppo-owned-shared-v1' and getattr(operations,'receipt',None) is not None:
        from gf_models.arrays import sha256
        from gf_training import hpo_gpu_lease as gpu
        stamp['termination_sha256']=sha256(gpu._GLOBAL_ROOT/'jobs'/stamp['job_id']/'termination.json')


def _identity(value: Any) -> str:
    if type(value) is not str or re.fullmatch('[0-9a-f]{32}',value) is None:raise ContractError('shared original identity required')
    return value


def _directories(scan: Any,path: Any,names: set[str]) -> None:
    if not names and not path.exists():return
    scan.directories(path,names)


def validate_inventory(registry: Json,state: Json,records: list[Json],scan: Any) -> dict[str,Json]:
    """Derive required service/job/checkpoint coverage without trusting refs to exist."""
    if (state.get('execution',{}).get('service',{}).get('record_version')!=2
            or type(state.get('invocations')) is not list):
        raise ContractError('shared invocation correspondence version2 required')
    refs=registry.get('service_references')
    if type(refs) is not dict:raise ContractError('shared reference inventory required')
    services: dict[str,Json]={}
    expected_tokens=set()
    position=0
    count=len(records)-1
    for index,invocation in enumerate(state['invocations']):
        stamp=closed(invocation.get('shared_service'),STAMP_FIELDS)
        integer(stamp['schema_version'],1,1)
        if type(stamp['finished']) is not bool:raise ContractError('shared invocation completion must be boolean')
        active=not stamp['finished']
        if active and (state['status']!='running' or index!=len(state['invocations'])-1 or stamp['end_generation'] is not None):
            raise ContractError('shared active invocation is not current running owner')
        start=integer(stamp['start_generation'],position,position)
        end=count if active else integer(stamp['end_generation'],start,count)
        if invocation['mode']!=('fresh' if index==0 else 'resume'):raise ContractError('shared invocation mode/order differs')
        position=end
        sid=stamp['service_id']
        if sid is None:
            if (end!=start or any(stamp[key] is not None for key in ('execution','job_id','registration_sha256','proof_sha256','termination_sha256','tail_usage'))
                    or stamp['publications']!=[]):raise ContractError('never-created service contradicts committed usage')
            continue
        _identity(sid);_identity(stamp['execution'])
        checksum(stamp['registration_sha256'])
        if sid in services:raise ContractError('shared service reused across invocations')
        folder=scan.root/'services'/sid
        registration=scan.object(folder/'registration.json')
        if digest(registration)!=stamp['registration_sha256']:raise ContractError('shared stamped registration differs')
        closed(registration,{'schema_version','kind','service_id','execution','job_id','run_id','config_sha256',
            'runtime_sha256','epoch','broker_pid','gpu_pid','device','libraries'})
        integer(registration['schema_version'],1,1)
        integer(registration['epoch'],1,1)
        expected={'schema_version':1,'kind':'shared-service-registration','service_id':sid,'execution':stamp['execution'],
            'job_id':stamp['job_id'],'run_id':state['run_id'],'config_sha256':state['config_sha256'],
            'runtime_sha256':digest(state['execution']['service']['runtime']),'epoch':1,'device':'cuda:0',
            'libraries':state['execution']['service']['libraries']}
        if any(registration[key]!=value for key,value in expected.items()):raise ContractError('shared invocation registration identity differs')
        for key in ('broker_pid','gpu_pid'):integer(registration[key],1)
        from gf_training import hpo_gpu_lease as gpu
        gpu._ticket(stamp['job_id'])
        previous={'requests':0,'publications':0}
        required: set[str]=set()
        allowed={records[g]['inference_sha256'] for g in set(records[start]['parent_controls']['snapshot_pool'])|{start}}
        used_tokens=set()
        usage_limits: dict[str,int]={}
        windows=[]
        for generation in range(start+1,end+1):
            update=read_hpo_update(scan.root, generation, state, records[generation]['update_sha256'], scan=scan)
            usage=checked_usage(cast(Json, update.get('service_usage')))
            if canonical(update['source_proof'].get('service_usage'))!=canonical(usage):raise ContractError('shared update/Source usage differs')
            neural={label:row['policy_options']['expected_manifest_digest'] for label,row in update['policy_bindings'].items()
                if row['model_id']=='mlp' and row['policy_options'] is not None}
            if {label:row['checkpoint'] for label,row in usage['bindings'].items()}!=neural:
                raise ContractError('shared usage omits or changes required neural policies')
            if usage['generation']!=generation-1:raise ContractError('shared collection generation differs')
            windows.append(usage)
            allowed.update(records[g]['inference_sha256'] for g in set(records[generation]['parent_controls']['snapshot_pool'])|{generation})
            if not active or generation<end:
                required.update(records[g]['inference_sha256'] for g in set(records[generation]['parent_controls']['snapshot_pool'])|{generation})
        if stamp['tail_usage'] is not None:
            if active:raise ContractError('active invocation claims final uncommitted tail')
            tail=checked_usage(stamp['tail_usage'])
            if tail['generation']!=end:raise ContractError('shared tail is not the uncommitted current collection')
            expected_tail={records[g]['inference_sha256'] for g in set(records[end]['parent_controls']['snapshot_pool'])|{end}}
            if {binding['checkpoint'] for binding in tail['bindings'].values()}!=expected_tail:
                raise ContractError('shared uncommitted tail omits required current/pool checkpoints')
            windows.append(tail)
        for usage in windows:
            for key in ('service_id','execution','job_id','registration_sha256'):
                if usage[key]!=stamp[key]:raise ContractError('shared collection belongs to another invocation/job')
            if (usage['config_sha256']!=state['config_sha256'] or usage['runtime_sha256']!=registration['runtime_sha256']
                    or usage['start']!=previous):raise ContractError('shared collection counter window gap or identity differs')
            previous=usage['end']
            for row in usage['bindings'].values():
                required.add(row['checkpoint']);used_tokens.add(row['token'])
                usage_limits[row['token']]=min(usage_limits.get(row['token'],usage['end']['publications']),usage['end']['publications'])
                if row['token'] not in refs or refs[row['token']]['binding']['service_id']!=sid or refs[row['token']]['binding']['inference_sha256']!=row['checkpoint']:
                    raise ContractError('shared committed usage has missing/wrong cache reference')
        publications=stamp['publications']
        if type(publications) is not list:raise ContractError('shared publication sequence required')
        tokens=set()
        published=set()
        for ordinal,row in enumerate(publications,1):
            closed(row,{'schema_version','ordinal','token','checkpoint','generation','ack_sha256'})
            integer(row['schema_version'],1,1);integer(row['ordinal'],ordinal,ordinal)
            _identity(row['token']);checksum(row['ack_sha256'])
            g=integer(row['generation'],0,count)
            if row['checkpoint']!=records[g]['inference_sha256'] or row['token'] in tokens:raise ContractError('shared publication checkpoint/token differs')
            tokens.add(row['token']);published.add(row['checkpoint'])
            if row['token'] not in refs or refs[row['token']]['binding']['service_id']!=sid or refs[row['token']]['binding']['generation']!=g:
                raise ContractError('shared publication lacks exact acquisition')
            actual=scan.object(folder/'references'/row['token']/'publication.json')
            ack={'accepted':True,'checkpoint':row['checkpoint'],'publication':ordinal}
            if actual!=row|{'ack':ack} or row['ack_sha256']!=digest(ack):raise ContractError('shared broker publication ordinal differs')
            if ordinal>usage_limits.get(row['token'],ordinal):
                raise ContractError('shared collection used an unpublished acquisition')
        proof=None
        if not active:
            if not required<=published or not published<=allowed or not used_tokens<=tokens or previous['publications']>len(publications):
                raise ContractError('shared publication inventory lacks required committed checkpoints')
            checksum(stamp['proof_sha256'])
            proof=scan.object(folder/'proof.json')
            retirement=scan.object(folder/'retirement.json')
            if digest(proof)!=stamp['proof_sha256'] or digest(retirement)!=proof['retirement_sha256']:
                raise ContractError('shared stamped final proof differs')
            closed(proof,{'key','service_id','broker_pid','broker_outcome','gpu_pid','gpu_outcome','gpu_exitcode','registration_sha256','retirement_sha256'})
            integer(proof['key'],1,1)
            integer(retirement.get('schema_version'),1,1)
            for key in ('broker_pid','gpu_pid'):
                integer(proof[key],1)
                integer(retirement.get(key),1)
            for key in ('gpu_exitcode','pending','active_relays','failures'):integer(retirement.get(key),0,0)
            integer(retirement.get('requests'),previous['requests'],previous['requests'])
            integer(retirement.get('publications'),len(publications),len(publications))
            if (proof['service_id']!=sid or proof['registration_sha256']!=digest(registration)
                    or proof['broker_pid']!=registration['broker_pid'] or proof['gpu_pid']!=registration['gpu_pid']
                    or proof['broker_outcome']!='reaped' or proof['gpu_outcome']!='reaped' or proof['gpu_exitcode']!=0
                    or type(proof['gpu_exitcode']) is not int or retirement.get('registration_sha256')!=digest(registration)
                    or retirement.get('successful') is not True or retirement.get('relays_joined') is not True
                    or retirement.get('accepted') is not True or retirement.get('kind')!='shared-service-retirement'
                    or retirement.get('broker_pid')!=registration['broker_pid']
                    or retirement.get('service_id')!=sid or retirement.get('gpu_pid')!=registration['gpu_pid']
                    or retirement.get('gpu_exitcode')!=0 or retirement.get('pending')!=0 or retirement.get('active_relays')!=0
                    or retirement.get('failures')!=0 or retirement.get('requests')!=previous['requests']
                    or retirement.get('publications')!=len(publications)):
                raise ContractError('shared invocation final retirement/counters differ')
            if state.get('run_variant')=='hpo-ppo-owned-shared-v1':
                original=gpu._GLOBAL_ROOT/'jobs'/stamp['job_id']
                from gf_models.arrays import sha256
                if sha256(original/'termination.json')!=checksum(stamp['termination_sha256']):raise ContractError('shared stamped global termination differs')
                admission=gpu._read(original/'admission.json')
                receipt=gpu._termination(gpu._read(original/'termination.json'),admission)
                if (not isinstance(receipt, gpu.TerminationReceipt) or receipt.schema_version!=2 or getattr(receipt,'services',())!=(proof,)
                        or admission['ticket']!=stamp['job_id'] or admission['context']['run_id']!=state['run_id']
                        or admission['context']['kind']!='training'):
                    raise ContractError('shared invocation lacks original schema2 training termination')
            elif stamp['termination_sha256'] is not None:raise ContractError('ordinary shared service cannot claim HPO global termination')
            service_tokens={token for token,row in refs.items() if row['binding']['service_id']==sid}
            if service_tokens!=tokens:raise ContractError('shared reference/publication inventory differs')
            _directories(scan,folder/'references',tokens)
        else:
            tokens.update(token for token,row in refs.items() if row['binding']['service_id']==sid)
        expected_tokens.update(tokens)
        services[sid]={'registration':registration,'proof':proof}
    if position!=count or (state['status']!='running' and not state['invocations']):raise ContractError('shared invocation coverage omits committed generations')
    if set(refs)!=expected_tokens:raise ContractError('shared cache references belong to no original invocation')
    _directories(scan,scan.root/'services',set(services))
    return services
