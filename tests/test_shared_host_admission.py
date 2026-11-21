"""Calibrated shared CPU ownership; logical caps and CUDA geometry stay intact."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common.domain import ContractError
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT


def configuration(games=16,batch=None):
    from gf_cli.hpo_config import load_hpo_config
    config=load_hpo_config(Path(__file__).parents[1]/'configs/hpo/mlp-shared-inference-scaling.toml').config
    operating=asdict(replace(config.operating,game_concurrency=games,max_participant_workers=2*games))
    runtime=config.runtime|{'max_batch_size':games if batch is None else batch}
    return config,operating,runtime


def admit(config,operating,runtime):
    from gf_models.shared_admission import shared_training_admission
    return shared_training_admission(config.model,config.inference,config.training.tbptt_length,operating,runtime,layout=LAYOUT)


def test_c16_fits_unchanged_globals_with_shape_based_calibrated_roles_and_phase_ownership():
    config,operating,runtime=configuration()
    value=admit(config,operating,runtime)
    assert value['host_recipe']=='shared-mlp-host-calibrated-v1'
    assert value['worker_host_bytes']==32*368377856
    assert value['host_roles']=={'participants_private_bytes':32*368377856,'shared_runtime_bytes':439175168,
        'service_private_bytes':1410199552,'broker_private_bytes':462737408,'coordinator_private_bytes':56020992,
        'tracker_private_bytes':6672384,'simulator_private_bytes':16*12427264}
    assert value['persistent_host_bytes']==17327842832
    assert value['broker_numeric_staging_bytes']==123873440
    assert value['publication_host_staging_bytes']==7746056
    assert value['ram_phases']=={'collection':23894167216,'update':20012197392,'save':21085939216,'sync':19483072536}
    assert value['peak_ram_bytes']<operating['managed_ram_bytes']==24*1024**3
    assert value['peak_cuda_bytes']==3951887330<operating['managed_cuda_bytes']==12*1024**3
    observed=value['host_calibration']['observed_occupancy']
    assert observed['component_checkpoints']==1 and observed['actual_concurrent_checkpoints']==2
    assert observed['configured_cache_entries']==6 and observed['filled_cache_measured'] is False
    assert value['host_calibration']['broker_python_framing_bound'] is False
    assert value['host_calibration']['shared_pss_source_overlap_possible'] is True
    assert 'not proven disjoint' in value['host_calibration']['shared_pss_scope']


def test_unused_participant_capacity_and_nested_parent_cap_are_not_extra_allocations():
    config,operating,runtime=configuration(4,2)
    value=admit(config,operating,runtime)
    changed=operating|{'max_participant_workers':32,'parent_budget_bytes':operating['parent_budget_bytes']+1024**2}
    after=admit(config,changed,runtime)
    assert after['workers']==value['workers']==8
    assert after['host_roles']==value['host_roles'] and after['ram_phases']==value['ram_phases']
    assert after['persistent_host_bytes']==value['persistent_host_bytes']
    assert after['host_limits']['parent_nested_in_source'] is True
    assert after['host_limits']['parent_budget_bytes']==changed['parent_budget_bytes']


@pytest.mark.parametrize('owner,phases',[('source_budget_bytes',{'collection','update','save','sync'}),
    ('collector_budget_bytes',{'collection'}),('numeric_budget_bytes',{'update','save'}),('save_budget_bytes',{'save'})])
def test_each_independent_owner_allowance_occurs_once_in_its_reachable_phases(owner,phases):
    config,operating,runtime=configuration(2)
    before=admit(config,operating,runtime)
    after=admit(config,operating|{owner:operating[owner]+4096},runtime)
    assert {name:after['ram_phases'][name]-amount for name,amount in before['ram_phases'].items()}=={
        name:4096 if name in phases else 0 for name in before['ram_phases']}


def test_broker_raw_staging_uses_games_even_when_batch_is_smaller_and_shared_backing_once():
    config,operating,runtime=configuration(8,2)
    value=admit(config,operating,runtime)
    service=value['service'];raw=value['host_staging']
    expected=8*(service['row_input_bytes']+service['row_output_bytes'])+service['largest_input_leaf_bytes']+8*service['largest_output_leaf_bytes']
    assert value['broker_numeric_staging_bytes']==expected
    assert raw['input_output_backings_bytes']==8*(service['row_input_bytes']+service['row_output_bytes'])
    assert raw['shared_backing_count']==1 and raw['decoded_json_framing_included'] is False
    assert service['effective_max_batch_size']==2


def test_host_cache_is_source_owned_and_gpu_weights_are_not_a_second_host_capacity():
    config,operating,runtime=configuration(2)
    before=admit(config,operating,runtime)
    model=replace(config.model,state=replace(config.model.state,hidden_sizes=(256,)),
        temporal=replace(config.model.temporal,hidden_size=256),
        heads=replace(config.model.heads,actor_hidden_sizes=(256,),critic_hidden_sizes=(256,)))
    after=admit(replace(config,model=model),operating,runtime)
    assert after['service']['parameter_bytes']!=before['service']['parameter_bytes']
    assert after['persistent_host_bytes']-after['learner_bytes']==before['persistent_host_bytes']-before['learner_bytes']
    assert after['publication_host_staging_bytes']==after['service']['parameter_bytes']+after['service']['position_bytes']
    assert after['host_calibration']['checkpoint_backing_owner']=='source_budget_bytes'
    assert after['host_calibration']['extrapolation']['model_payload_differs'] is True


@pytest.mark.parametrize('resource',['ram','cuda'])
def test_calibrated_formula_keeps_exact_managed_global_refusal(resource):
    config,operating,runtime=configuration(2)
    value=admit(config,operating,runtime)
    limit=value['peak_'+resource+'_bytes']
    admitted=admit(config,operating|{'managed_'+resource+'_bytes':limit},runtime)
    assert admitted['peak_'+resource+'_bytes']==limit
    with pytest.raises(ContractError,match='resource='+resource):
        admit(config,operating|{'managed_'+resource+'_bytes':limit-1},runtime)


def test_logical_worker_and_checkpoint_inventory_refusals_are_unchanged():
    config,operating,runtime=configuration(2)
    with pytest.raises(ContractError,match='concurrent roles'):
        admit(config,operating|{'max_participant_workers':3},runtime)
    with pytest.raises(ContractError,match='frozen opponent inventory'):
        admit(config,operating|{'opponent_pool_size':6},runtime)
