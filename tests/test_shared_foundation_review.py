"""Reproductions of the independent foundation review's two boundary findings."""
from copy import deepcopy
from pathlib import Path

import pytest

from gf_common.domain import ContractError, digest


@pytest.mark.parametrize('schema',[1,2,3,4,5])
def test_hpo_rejects_ordinary_shared_serving_base_for_every_campaign_mode(tmp_path,schema):
    from gf_cli.hpo_config import campaign_config
    from tests.test_hpo_config import campaign_raw
    raw=campaign_raw()
    raw.update(schema_version=schema,base_experiment=str(Path(__file__).parents[1]/'configs/experiments/mlp-shared-inference-smoke.toml'))
    if schema==2:
        raw.update(training={'collection_games':16},training_opponents={'version':'1','preset':'mixed-refresh-16-v1',
            'refresh':{'version':'1','attack_threshold':12}})
    if schema>=3:raw['runtime']={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728}
    if schema>=4:raw.update(origin='producer-owned-v1',encoder='canonical-python')
    if schema==5:raw['runtime']={'mode':'shared-mlp-v1'}
    with pytest.raises(ContractError,match='base.*inference_service'):
        campaign_config(raw,origin=tmp_path)


def cpu_retirement():
    from gf_common.shared_inference import CPU_PROTOCOL, cpu_reset_fields
    from tests.hpo_resident_support import retirement_fact
    value=retirement_fact()
    value.update(schema_version=2,kind='hpo-shared-cpu-logical-retirement',
        service_quiescence={'service_id':'e'*32,'pending':0,'publications':1,'requests':1})
    for row in value['owners']:
        for name in ('end_ack','park_ack'):
            ack=row[name]
            del ack['cuda_synchronized']
            ack.update(schema_version=2,protocol=CPU_PROTOCOL,**cpu_reset_fields())
            row[name+'_sha256']=digest(ack)
    return value


@pytest.mark.parametrize('key,value',[('job_ticket','f'*32),('pool_id','f'*32),('pool_epoch',99)])
def test_cpu_retirement_rejects_forged_top_level_owner_identity(key,value):
    from gf_common.shared_inference import checked_cpu_retirement
    original=cpu_retirement()
    assert checked_cpu_retirement(original,max_workers=4)==original
    with pytest.raises(ContractError):checked_cpu_retirement(original|{key:value},max_workers=4)


def test_cpu_park_checkpoint_must_match_same_episode_end_even_with_valid_ack_hash():
    from gf_common.shared_inference import checked_cpu_retirement
    value=cpu_retirement()
    row=value['owners'][0]
    row['park_ack']['checkpoint_sha256']='f'*64
    row['park_ack_sha256']=digest(row['park_ack'])
    with pytest.raises(ContractError):checked_cpu_retirement(value,max_workers=4)


@pytest.mark.parametrize('field',['clean_reset','private_state_empty','pending_io_empty','cpu_only'])
def test_cpu_reset_receipt_requires_exact_boolean_true(field):
    from gf_common.shared_inference import checked_cpu_ack
    ack=deepcopy(cpu_retirement()['owners'][0]['park_ack'])
    ack[field]=1
    with pytest.raises(ContractError):checked_cpu_ack(ack)
