"""Bounded experimental update planning and numerical mechanisms."""
from dataclasses import replace

import pytest

torch = pytest.importorskip('torch')

from gf_common.domain import ContractError
from gf_training.hpo_training_execution import (
    TrainingExecutionSpec,
    read_execution_spec,
    write_execution_spec,
)
from gf_training.hpo_update_planning import plan_groups
from tests.test_hpo_numeric import learner_for, rollout_for


def test_closed_spec_and_hashed_file(tmp_path):
    spec = TrainingExecutionSpec(variant='microbatch', diagnostic_directory=str(tmp_path/'diagnostics'))
    path = tmp_path/'settings.json'
    sha = write_execution_spec(path, spec)
    assert read_execution_spec(path, sha) == spec
    with pytest.raises(ContractError):
        read_execution_spec(path, '0'*64)
    with pytest.raises(ContractError):
        TrainingExecutionSpec(variant='unknown')
    with pytest.raises(ContractError):
        write_execution_spec(path, spec)


@pytest.mark.parametrize('count,batch', [(53,48),(5,48),(47,48),(7,48)])
def test_planner_preserves_endpoint_membership_and_logical_tails(count,batch):
    learner = learner_for()
    rollout = rollout_for(learner, count=count)
    order = list(reversed(range(count)))
    groups = []
    for start in range(0,count,batch):
        logical = order[start:start+batch]
        planned = plan_groups(rollout, logical, learner.model, learner.binding,
            learner.training.tbptt_length, variant='microbatch', reservation_bytes=learner.binding.managed_budget_bytes,
            layout=learner.network.layout)
        assert sorted(i for group in planned for i in group.indices) == sorted(logical)
        assert sum(len(group.indices)/len(logical) for group in planned) == pytest.approx(1)
        assert all(len(group.indices)<=16 for group in planned)
        groups.extend(planned)
    assert len([i for g in groups for i in g.indices]) == count


@pytest.mark.parametrize('kind', ['mlp','gru','transformer'])
@pytest.mark.parametrize('variant', ['metrics','packed','microbatch','shared-prefix'])
def test_experimental_update_matches_baseline_parameters_and_adam(kind,variant):
    if kind=='mlp' and variant=='shared-prefix':
        pytest.skip('sequence only')
    from gf_training.hpo_config import HPOTrainingConfig
    recipe = replace(HPOTrainingConfig(), epochs=1, minibatch_size=5)
    baseline = learner_for(kind, recipe=recipe)
    candidate = learner_for(kind, recipe=recipe)
    rollout = rollout_for(baseline, count=7)
    candidate._training_execution = TrainingExecutionSpec(variant=variant)
    candidate._training_reservation_bytes = candidate.binding.managed_budget_bytes
    expected = baseline._update_numeric(rollout, {}, 0., baseline.network)
    actual = candidate._update_numeric(rollout, {}, 0., candidate.network)
    assert expected['optimizer_step_delta'] == actual['optimizer_step_delta'] == 2
    assert actual['endpoint_evaluations']==7
    for left,right in zip(baseline.network.parameters(),candidate.network.parameters(),strict=True):
        assert (left.grad is None)==(right.grad is None)
        assert (left in baseline.optimizer.state)==(right in candidate.optimizer.state)
        if left in baseline.optimizer.state:
            assert baseline.optimizer.state[left]['step']==candidate.optimizer.state[right]['step']
            assert all(torch.isfinite(value).all() for value in candidate.optimizer.state[right].values())
    for key in ('total_loss','value_mse','policy_loss','entropy','approx_kl'):
        torch.testing.assert_close(torch.tensor(actual[key]),torch.tensor(expected[key]))


@pytest.mark.parametrize('kind', ['mlp','gru','transformer'])
def test_packed_batch_gradients_match_endpoint_sum(kind):
    from gf_models.structured_learning import pack_window
    from gf_training.hpo_update_execution import endpoint_terms
    from gf_training.hpo_update_planning import group_frames
    baseline=learner_for(kind)
    candidate=learner_for(kind)
    rollout=rollout_for(baseline,count=3)
    # Repeat the same endpoint to exercise real tensor batching at equal lengths.
    endpoint=rollout.endpoint(2)
    vectors=[rollout.vector(2)]*3
    for _ in range(3):
        logs,value,_=baseline.network(*endpoint,suffix_length=2)
        selected,old,_,advantage,target=vectors[0]
        loss,_=endpoint_terms(logs[None],value[None],torch.ones(1,len(logs),dtype=torch.bool),
            torch.tensor([selected]),*[torch.tensor([v]) for v in (old,advantage,target)],training=baseline.training)
        (loss.mean()/3).backward()
    frames=group_frames([endpoint]*3,candidate.model,False)
    window=pack_window(candidate.network,frames,[endpoint[1]]*3,vectors)
    logs,values=candidate.network._learning_forward(window,[3]*3,2)
    loss,_=endpoint_terms(logs,values,window.operands['mask'],window.operands['selected'],
        *window.operands['vectors'],training=candidate.training)
    loss.mean().backward()
    for left,right in zip(baseline.network.parameters(),candidate.network.parameters(),strict=True):
        assert (left.grad is None)==(right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad,right.grad)


def test_admission_rejects_before_packing(monkeypatch):
    from gf_training import hpo_update_execution
    learner=learner_for()
    rollout=rollout_for(learner)
    learner._training_execution=TrainingExecutionSpec(variant='packed')
    learner._training_reservation_bytes=1
    monkeypatch.setattr(hpo_update_execution,'pack_window',lambda *args:pytest.fail('allocated before admission'))
    with pytest.raises(ContractError,match='reservation'):
        learner._update_numeric(rollout,{},0.,learner.network)
    assert learner.steps==0 and not learner.optimizer.state


def test_shared_prefix_requires_matching_actual_frame_origin():
    from gf_training.hpo_update_planning import plan_groups
    learner=learner_for('gru')
    rollout=rollout_for(learner,count=4)
    groups=plan_groups(rollout,list(range(4)),learner.model,learner.binding,32,
        variant='shared-prefix',reservation_bytes=learner.binding.managed_budget_bytes,layout=learner.network.layout)
    assert any(group.shared for group in groups)
    assert sum(len(g.indices) for g in groups)==4
    # Same textual episode, distinct actual tokens must not merge.
    import copy
    real=rollout.endpoint
    class SeparateOrigins:
        resident_bytes=rollout.resident_bytes
        def endpoint(self,index):
            frames,candidates=real(index)
            return tuple(copy.deepcopy(frame) for frame in frames),candidates
    groups=plan_groups(SeparateOrigins(),list(range(4)),learner.model,learner.binding,32,
        variant='shared-prefix',reservation_bytes=learner.binding.managed_budget_bytes,layout=learner.network.layout)
    assert not any(group.shared for group in groups)


def test_padding_is_excluded_from_normalization_loss_and_checkpoint_backward():
    from gf_models.structured_learning import pack_window
    from gf_training.hpo_update_execution import endpoint_terms
    learner=learner_for()
    rollout=rollout_for(learner,count=3)
    frames,candidates=rollout.endpoint(2)
    # Add a real smaller candidate table from the same actor frame.
    from dataclasses import replace
    small=replace(candidates,kinds=candidates.kinds[:1],targets=candidates.targets[:1],
        numerics=candidates.numerics[:1],present=candidates.present[:1],
        offsets=(0,candidates.offsets[1]),slots=candidates.slots[:candidates.offsets[1]])
    vector=rollout.vector(2)
    window=pack_window(learner.network,[frames[-1]]*2,[candidates,small],
        [vector,(0,*vector[1:])])
    saved=tuple(value.clone() for value in window.storage)
    logs,values=learner.network._learning_forward(window,[3,3],3)
    assert logs[1,0]==0 and torch.isneginf(logs[1,1:]).all()
    loss,_=endpoint_terms(logs,values,window.operands['mask'],window.operands['selected'],
        *window.operands['vectors'],training=learner.training)
    loss.mean().backward()
    assert all(torch.equal(a,b) for a,b in zip(saved,window.storage,strict=True))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in learner.network.parameters())


def test_nonfinite_real_endpoint_cannot_be_hidden_by_reduction():
    from gf_training.hpo_update_execution import endpoint_terms
    learner=learner_for()
    with pytest.raises(ContractError,match='endpoint loss'):
        endpoint_terms(torch.zeros(2,1),torch.tensor([0.,float('inf')]),
            torch.ones(2,1,dtype=torch.bool),torch.zeros(2,dtype=torch.int64),
            torch.zeros(2),torch.ones(2),torch.zeros(2),training=learner.training)
