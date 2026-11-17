from dataclasses import replace

"""Normal Model4 process caps use original creators and distinct CPU/GPU ownership."""

import pytest

from gf_common.domain import ContractError
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.hpo_learning_process import HpoLearningProcess
from gf_training.ppo_run import LearningProcess
from tests.test_card_instance_learning import ACTIONS
from tests.test_hpo_numeric import learner_for


def process_for(tmp_path,*,owned=False):
    learner=learner_for()
    options={}
    if owned:
        from gf_training.hpo_owned_process import HpoOwnedLearningProcess
        from tests.test_hpo_owned_numeric import budget
        cls=HpoOwnedLearningProcess
        options['budget']=budget()
    else:
        cls=HpoLearningProcess
    process=cls(tmp_path,1024**2,learner.model,learner.training,learner.binding,
        learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ReservationLedger(),
        cuda_visible_devices='',cleanup_deadline=float('inf'),**options)
    return process,learner


def test_frozen_base_creator_launch_hook_is_empty(tmp_path):
    assert LearningProcess(tmp_path,1024)._launch_arguments()==()


@pytest.mark.parametrize('owned',[False,True])
def test_normal_cpu_cap_is_positive_while_same_gpu_lease_remains_zero(tmp_path,owned):
    process,learner=process_for(tmp_path,owned=owned)
    process.lease=process.ledger.reserve_learner(process.resident_bytes,process)
    process.lease.creation_pending=True
    assert process.lease.resident_bytes==0
    args=process._launch_arguments()
    assert args==('--model4-execution-cap',str(learner._training_reservation_bytes))
    process.lease.creation_pending=False
    process.lease.cancel_before_spawn()
    if owned:
        process.channel.close();process._inherited.close()


def test_cap_requires_same_original_active_creation_lease(tmp_path):
    process,_=process_for(tmp_path)
    with pytest.raises(ContractError,match='SAME|active'):
        process._launch_arguments()
    process.lease=process.ledger.reserve_learner(0,process)
    process.lease.creation_pending=True
    process.lease._cleanup_owner=object()
    with pytest.raises(ContractError,match='SAME|active'):
        process._launch_arguments()
    assert not process._creation_no_child
    process.lease._cleanup_owner=process
    process.lease.creation_pending=False
    process.lease.cancel_before_spawn()


def test_execution_cap_parser_is_closed_positive_and_canonical():
    from gf_training.hpo_learning_process import parse_execution_cap
    assert parse_execution_cap(()) is None
    assert parse_execution_cap(('--model4-execution-cap','123'))==123
    for args in (('123',),('--other','123'),('--model4-execution-cap','0'),
            ('--model4-execution-cap','+123'),('--model4-execution-cap','0123'),
            ('--model4-execution-cap','123','extra')):
        with pytest.raises(ContractError):
            parse_execution_cap(args)


def test_gpu_cap_is_exact_same_active_lease_without_gpu_allocation(tmp_path):
    learner=learner_for()
    binding=replace(learner.binding,device='cuda:0')
    process=HpoLearningProcess(tmp_path,1024**2,learner.model,learner.training,binding,
        learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ReservationLedger(),
        cuda_visible_devices='0',cleanup_deadline=float('inf'))
    process.lease=process.ledger.reserve_learner(process.resident_bytes,process)
    process.lease.creation_pending=True
    assert process._launch_arguments()==('--model4-execution-cap',str(process.lease.resident_bytes))
    actual=process.lease.resident_bytes
    process.lease.resident_bytes=actual-1
    with pytest.raises(ContractError,match='SAME reserved'):
        process._launch_arguments()
    assert not process._creation_no_child
    process.lease.resident_bytes=actual
    process.lease.creation_pending=False
    process.lease.cancel_before_spawn()


def test_cap_validation_failure_never_invents_stdlib_no_child_proof(tmp_path,monkeypatch):
    import asyncio
    process,_=process_for(tmp_path)
    async def forbidden(*args,**kwargs):
        pytest.fail('OS creator before cap/lease validation')
    monkeypatch.setattr(asyncio,'create_subprocess_exec',forbidden)
    with pytest.raises(ContractError,match='SAME active'):
        asyncio.run(process._create_process())
    assert not process._creation_no_child and process.process is None


def test_non_hpo_worker_rejects_labeled_cap_before_model_allocation(monkeypatch):
    from gf_training import structured_numeric_worker
    monkeypatch.setattr(structured_numeric_worker,'model_config_v2',lambda *args:pytest.fail('allocated older model'))
    initial={'protocol_version':3,'command':'initialize','identity':{},'model':{},'seed':0,'output':'unused','restore':None}
    with pytest.raises(ContractError,match='older/frozen'):
        structured_numeric_worker.serve(initial,(),execution_cap=1)


def test_missing_or_wrong_model4_cap_fails_closed():
    from gf_training.hpo_learning_process import checked_execution_cap
    learner=learner_for()
    for cap in (None,0,learner._training_reservation_bytes-1,learner.binding.managed_budget_bytes+1):
        with pytest.raises(ContractError):
            checked_execution_cap(learner.model,learner.training,learner.binding,cap)


@pytest.mark.parametrize('fault',['missing','invalid'])
def test_bad_optional_diagnostic_setup_keeps_real_owned_update_and_reap(tmp_path,monkeypatch,fault):
    from tests.test_learning_update_telemetry import _owned_observation_lifecycle
    original=HpoLearningProcess._environment
    def environment(self):
        value=original(self)
        value['GFB_HPO_DIAGNOSTIC_INVOCATION']='optional-setup-fault'
        if fault=='missing':
            value.pop('GFB_HPO_DIAGNOSTIC_LOG_BYTES',None)
        else:
            value['GFB_HPO_DIAGNOSTIC_LOG_BYTES']='invalid'
        return value
    # Test-only fault injection; production parent environment remains minimal.
    monkeypatch.setattr(HpoLearningProcess,'_environment',environment)
    _owned_observation_lifecycle(tmp_path,'off')
