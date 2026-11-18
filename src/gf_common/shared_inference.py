"""Closed shared-serving controls, never features or live ownership authority."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Self

from gf_common.domain import ContractError, Json, closed, integer

SHARED_MODE = 'shared-mlp-v1'
SHARED_VARIANT = 'hpo-ppo-owned-shared-v1'
SHARED_ADMISSION = 'shared-mlp-admission-v1'
CPU_PROTOCOL = 'shared-cpu-worker-v1'
SHARED_LIBRARIES = MappingProxyType({'torchrl':'0.15.0','tensordict':'0.15.2'})


def shared_dependency_versions() -> dict[str,str]:
    """Read optional distribution metadata without importing either runtime."""
    from importlib.metadata import PackageNotFoundError, version
    try:actual={name:version(name) for name in SHARED_LIBRARIES}
    except PackageNotFoundError as error:
        raise ContractError('shared-inference optional dependencies are not installed') from error
    if actual!=dict(SHARED_LIBRARIES):raise ContractError('unverified shared-inference runtime versions')
    return actual


def shared_runtime(value: Json, *, game_concurrency: int) -> Json:
    raw = closed(value, {'mode'}, {'max_batch_size','formation_wait_ms','cache_entries','cache_weight_bytes'})
    if raw['mode'] != SHARED_MODE:
        raise ContractError('explicit shared-mlp-v1 serving required')
    result = {'mode':SHARED_MODE,'max_batch_size':game_concurrency,'formation_wait_ms':1.0,
        'cache_entries':6,'cache_weight_bytes':134217728} | raw
    integer(result['max_batch_size'],1,2**63-1)
    integer(result['cache_entries'],6,6)
    integer(result['cache_weight_bytes'],134217728,134217728)
    if type(result['formation_wait_ms']) not in (float,int) or result['formation_wait_ms'] != 1.0:
        raise ContractError('shared formation wait must be1ms')
    return result


def runtime_for_spec(spec: Json) -> Json:
    if spec.get('schema_version')==5:
        return shared_runtime(spec['runtime'],game_concurrency=spec['operating']['game_concurrency'])
    from gf_common.hpo_resident import resident_runtime
    return resident_runtime(spec['runtime'])


def cpu_reset_fields() -> Json:
    return {'clean_reset':True,'private_state_empty':True,'pending_io_empty':True,
        'cpu_only':True,'reset_version':'shared-cpu-reset-v1'}


def checked_cpu_ack(value: Json) -> Json:
    from gf_common.hpo_resident import CACHE_OWNER_KEYS, REQUEST_KEYS, _sha, cache_owner
    raw=closed(value,REQUEST_KEYS|set(cpu_reset_fields()))
    cache_owner({key:raw[key] for key in CACHE_OWNER_KEYS})
    integer(raw['schema_version'],2,2)
    integer(raw['generation'],0,2**63-1)
    integer(raw['request_epoch'],1,2**63-1)
    _sha(raw['checkpoint_sha256'])
    if raw['cache_identity'] is not None:_sha(raw['cache_identity'])
    if (raw['protocol']!=CPU_PROTOCOL or raw['command'] not in ('END','PARK','SYNC','EVICT')
            or type(raw['run_id']) is not str or not raw['run_id'] or len(raw['run_id'])>128
            or any(type(raw[key]) is not type(expected) or raw[key]!=expected for key,expected in cpu_reset_fields().items())):
        raise ContractError('explicit original CPU actor drain acknowledgement required')
    if (raw['command']=='EVICT')!=(raw['cache_identity'] is not None):
        raise ContractError('CPU eviction requires exact metadata cache identity')
    return raw


def checked_cpu_retirement(value: Any,*,max_workers: int) -> Json:
    import re

    from gf_common.domain import digest
    from gf_common.hpo_resident import CACHE_OWNER_KEYS, cache_owner
    raw=closed(value,{'schema_version','kind','run_id','collection_id','job_ticket','pool_id','pool_epoch',
        'generation','owners','service_quiescence'})
    integer(raw['schema_version'],2,2)
    integer(max_workers,1,2**63-1)
    integer(raw['pool_epoch'],1,2**63-1)
    integer(raw['generation'],0,2**63-1)
    if (raw['kind']!='hpo-shared-cpu-logical-retirement' or type(raw['owners']) is not list or not 1<=len(raw['owners'])<=max_workers
            or type(raw['collection_id']) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',raw['collection_id']) is None):
        raise ContractError('shared CPU collection retirement required')
    quiescence=closed(raw['service_quiescence'],{'service_id','pending','publications','requests'})
    integer(quiescence['pending'],0,0)
    integer(quiescence['publications'],0,2**63-1)
    integer(quiescence['requests'],0,2**63-1)
    if type(quiescence['service_id']) is not str or re.fullmatch('[0-9a-f]{32}',quiescence['service_id']) is None:
        raise ContractError('CPU drain lacks original service identity')
    seen=set()
    viewers={1:0,2:0}
    for row in raw['owners']:
        closed(row,CACHE_OWNER_KEYS|{'end_ack','end_ack_sha256','park_ack','park_ack_sha256'})
        cache_owner({key:row[key] for key in CACHE_OWNER_KEYS})
        if any(row[key]!=raw[key] for key in ('job_ticket','pool_id','pool_epoch')):
            raise ContractError('CPU retirement has foreign job/pool identity')
        if row['worker_key'] in seen:raise ContractError('CPU retirement repeats physical owner')
        seen.add(row['worker_key']);viewers[row['viewer']]+=1
        if viewers[row['viewer']]>max_workers//2:raise ContractError('CPU viewer partition exceeds original capacity')
        for name,command in (('end_ack','END'),('park_ack','PARK')):
            ack=checked_cpu_ack(row[name])
            if ack['command']!=command or row[name+'_sha256']!=digest(ack):raise ContractError('CPU drain acknowledgement differs')
            for key in CACHE_OWNER_KEYS|{'run_id','generation'}:
                if ack[key]!=(row[key] if key in CACHE_OWNER_KEYS else raw[key]):raise ContractError('CPU drain owner/epoch differs')
        if (row['park_ack']['request_epoch']<=row['end_ack']['request_epoch']
                or row['park_ack']['checkpoint_sha256']!=row['end_ack']['checkpoint_sha256']):
            raise ContractError('CPU PARK must follow same-checkpoint original END')
    return raw


def serving_config(config: object) -> Json | None:
    spec = getattr(config,'hpo_spec',{})
    if spec.get('schema_version') == 5:
        return shared_runtime(spec['runtime'],game_concurrency=spec['operating']['game_concurrency'])
    if hasattr(config,'hpo_spec'):return None
    run = getattr(config,'run',None)
    value = getattr(run,'inference_service',None)
    if value is None:return None
    assert run is not None
    return shared_runtime(value,game_concurrency=run.concurrency)


def evaluation_runtime(config: object) -> Json | None:
    """Training identity stays unchanged; schema5 evaluates through fresh actors."""
    if getattr(config,'schema_version',None)==5:return None
    runtime=getattr(config,'runtime',None)
    if runtime is None:return None
    from gf_common.hpo_resident import resident_runtime
    return resident_runtime(runtime)


class ServingBudget:
    """Bounded serving bank/staging owner, without a training dependency."""
    def __init__(self,limit: int) -> None:
        import threading
        self.limit,self.used,self.peak=limit,0,0
        self._lock=threading.Lock()

    def reserve(self,size: int,*,wait: bool = False) -> ServingReservation:
        integer(size,0,self.limit)
        with self._lock:
            if wait or self.used+size>self.limit:raise ContractError('shared serving host reservation exceeded')
            self.used+=size
            self.peak=max(self.peak,self.used)
        return ServingReservation(self,size)


class ServingReservation:
    def __init__(self,owner: ServingBudget,size: int) -> None:
        self.owner,self.size=owner,size

    def close(self) -> None:
        with self.owner._lock:
            self.owner.used-=self.size
            self.size=0

    def __enter__(self) -> Self:return self

    def __exit__(self,*args: object) -> None:self.close()


def checked_usage(value: Json) -> Json:
    import re

    from gf_common.hpo_resident import _sha
    raw=closed(value,{'schema_version','service_id','execution','job_id','registration_sha256','config_sha256',
        'runtime_sha256','generation','batch_id','start','end','bindings'})
    integer(raw['schema_version'],1,1)
    integer(raw['generation'],0)
    if raw['batch_id']!=f"batch-{raw['generation']:06}":raise ContractError('shared usage batch/generation differs')
    for key in ('registration_sha256','config_sha256','runtime_sha256'):_sha(raw[key])
    for key in ('start','end'):
        closed(raw[key],{'requests','publications'})
        for counter in ('requests','publications'):integer(raw[key][counter],0)
    if any(raw['end'][key]<raw['start'][key] for key in raw['start']):raise ContractError('shared counters moved backwards')
    if type(raw['bindings']) is not dict:raise ContractError('shared required bindings missing')
    for row in raw['bindings'].values():
        closed(row,{'checkpoint','token'})
        _sha(row['checkpoint'])
        if type(row['token']) is not str or re.fullmatch('[0-9a-f]{32}',row['token']) is None:
            raise ContractError('shared acquisition token required')
    return raw
