"""Automatic Model4 route and preplanned practical capacity; no Source authority."""
from dataclasses import replace

import pytest

torch=pytest.importorskip('torch')

from gf_common.domain import ContractError
from gf_common.structured_tokens import validate_candidate_tokens, validate_frame_tokens
from gf_training.hpo_config import HPOTrainingConfig
from gf_training.hpo_training_execution import TrainingExecutionSpec
from tests.test_card_instance_learning import ACTIONS, RESUME, RULES
from tests.test_hpo_numeric import learner_for, rollout_for


@pytest.mark.parametrize('kind,variant',[('mlp','microbatch'),('gru','shared-prefix'),('transformer','shared-prefix')])
def test_normal_model4_selects_optimized_route_without_experimental_authority(kind,variant):
    learner=learner_for(kind,recipe=replace(HPOTrainingConfig(),epochs=1))
    assert learner._training_execution is None
    assert learner._training_reservation_bytes>0
    rollout=rollout_for(learner,count=3)
    result=learner._update_numeric(rollout,{},0.,learner.network)
    assert learner._training_update_diagnostics['variant']==variant
    assert learner._training_execution is None and result['optimizer_step_delta']==1


def test_private_reference_still_selects_original_loop(monkeypatch):
    learner=learner_for(recipe=replace(HPOTrainingConfig(),epochs=1))
    learner._training_execution=TrainingExecutionSpec(variant='baseline')
    rollout=rollout_for(learner,count=3)
    monkeypatch.setattr(learner,'_update_experimental',lambda *args:pytest.fail('reference entered optimized loop'))
    result=learner._update_numeric(rollout,{},0.,learner.network)
    assert result['endpoint_evaluations']==3 and not hasattr(learner,'_training_update_diagnostics')


def controlled_endpoint(learner,length=1024,burst=0,burst_at=None):
    # Preserve all original token shape/category/missingness validators and own
    # sequence continuity, while keeping the controlled observation state fixed.
    frames,table=rollout_for(learner,count=1).endpoint(0)
    base=frames[0]
    history=[]
    event_end=0
    for index in range(length):
        events=burst if index==(length//2 if burst_at is None else burst_at) else 0
        start=event_end
        event_end+=events
        current=replace(base,revision=index,event_start=start,event_end=event_end,
            interval_numerics=tuple(base.interval_numerics[0] for _ in range(events)),
            interval_present=tuple(base.interval_present[0] for _ in range(events)),
            interval_categories=tuple(base.interval_categories[0] for _ in range(events)),
            history=(0,)*len(base.history))
        validate_frame_tokens(current,layout=learner.network.layout)
        history.append(current)
    validate_candidate_tokens(table,history[-1],learner.model.max_positions,layout=learner.network.layout)
    return tuple(history),table


@pytest.mark.parametrize('kind',['gru','transformer'])
def test_default32_long1024_prefix_is_streamed_before_allocation(kind):
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_update_planning import execution_plan
    tiny=learner_for(kind)
    model=replace(tiny.model,hand=replace(tiny.model.hand,attention_feedforward=1024))
    learner=HpoLearner(model,replace(HPOTrainingConfig(),epochs=1),17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    endpoint=controlled_endpoint(learner)
    plan=execution_plan([endpoint],model,learner.binding,32,layout=learner.network.layout,
        reservation_bytes=learner._training_reservation_bytes,resident_bytes=0)
    assert plan.staging is not None
    assert max(plan.estimate['tensor_bytes'].values())<=model.resources.max_tensor_bytes
    assert len(plan.staging.prefix_chunks)>1
    assert sum(len(rows) for rows in plan.staging.prefix_chunks)==992
    assert sum(len(rows) for rows in plan.staging.suffix_chunks)==32


def test_ragged_events_have_bounded_real_length_windows_and_no_full_slab():
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_update_planning import execution_plan
    tiny=learner_for('gru')
    model=replace(tiny.model,event=replace(tiny.model.event,max_interval=256,max_episode=2048))
    learner=HpoLearner(model,replace(HPOTrainingConfig(),epochs=1),17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    endpoint=controlled_endpoint(learner,burst=128)
    plan=execution_plan([endpoint],model,learner.binding,32,layout=learner.network.layout,
        reservation_bytes=learner._training_reservation_bytes,resident_bytes=0)
    assert plan.staging is not None
    frames=endpoint[0]
    assert all(len({len(frames[index].interval_numerics) for index in chunk})==1
        for chunk in (*plan.staging.prefix_chunks,*plan.staging.suffix_chunks))
    assert max(plan.estimate['tensor_bytes'].values())<=model.resources.max_tensor_bytes


def test_extreme_retained_graph_rejects_before_operand_allocation(monkeypatch):
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_update_planning import UpdateCapacityError, execution_plan
    tiny=learner_for('gru')
    model=replace(tiny.model,hand=replace(tiny.model.hand,attention_feedforward=1024,attention_heads=1,attention_layers=1))
    learner=HpoLearner(model,replace(HPOTrainingConfig(),epochs=1,tbptt_length=1024),17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    endpoint=controlled_endpoint(learner)
    with pytest.raises(UpdateCapacityError,match='retained|reservation|capacity'):
        execution_plan([endpoint],model,learner.binding,1024,layout=learner.network.layout,
            reservation_bytes=learner._training_reservation_bytes,resident_bytes=0)
    assert learner.steps==0 and not learner.optimizer.state


@pytest.mark.parametrize('kind',['gru','transformer'])
@pytest.mark.parametrize('burst',[0,128])
def test_real_streamed_long_prefix_outputs_gradients_and_input_lifetime(kind,burst,monkeypatch):
    import weakref

    from gf_models import structured_learning
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_update_execution import group_forward
    from gf_training.hpo_update_planning import UpdateGroup, execution_plan
    tiny=learner_for(kind)
    model=replace(tiny.model,hand=replace(tiny.model.hand,attention_feedforward=1024),
        event=replace(tiny.model.event,max_interval=256,max_episode=2048))
    recipe=replace(HPOTrainingConfig(),epochs=1)
    reference=HpoLearner(model,recipe,17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    candidate=HpoLearner(model,recipe,17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    endpoint=controlled_endpoint(reference,burst=burst,burst_at=1000)
    class Input:
        resident_bytes=0
        def endpoint(self,index):
            return endpoint
        def vector(self,index):
            return (0,-.5,0.,1.,.1)
    plan=execution_plan([endpoint],model,candidate.binding,32,layout=candidate.network.layout,
        reservation_bytes=candidate._training_reservation_bytes,resident_bytes=0)
    assert plan.staging is not None
    saved=[]
    original=structured_learning.pack_frames
    def capture(*args,**kwargs):
        window=original(*args,**kwargs)
        saved.append((torch.is_grad_enabled(),weakref.ref(window)))
        return window
    monkeypatch.setattr(structured_learning,'pack_frames',capture)
    logs,value,_=reference.network(*endpoint,suffix_length=32)
    from gf_training.hpo_update_execution import endpoint_terms
    expected,_=endpoint_terms(logs[None],value[None],torch.ones(1,len(logs),dtype=torch.bool),
        torch.tensor([0]),torch.tensor([-.5]),torch.tensor([1.]),torch.tensor([.1]),training=recipe)
    expected.mean().backward()
    loss,_metrics,window,_extra,encoded,copies=group_forward(candidate,Input(),UpdateGroup((0,),False,plan),
        TrainingExecutionSpec(variant='microbatch'),candidate.network)
    torch.testing.assert_close(loss,expected)
    assert encoded==1024 and copies>0
    assert all(ref() is None for gradient,ref in saved if not gradient)
    assert all(ref() is not None for gradient,ref in saved if gradient)
    snapshots=[tensor.clone() for retained in window.retained for tensor in retained.storage]
    loss.mean().backward()
    assert all(torch.equal(left,right) for left,right in zip(snapshots,
        (tensor for retained in window.retained for tensor in retained.storage),strict=True))
    for left,right in zip(reference.network.parameters(),candidate.network.parameters(),strict=True):
        assert (left.grad is None)==(right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad,right.grad)


def test_capacity_planner_does_not_retry_dead_or_domain_inputs():
    from gf_training.hpo_update_planning import plan_groups
    learner=learner_for()
    class Dead:
        resident_bytes=0
        def endpoint(self,index):
            raise ContractError('actual dead input')
    with pytest.raises(ContractError,match='actual dead'):
        plan_groups(Dead(),[0],learner.model,learner.binding,32,variant='microbatch',
            reservation_bytes=learner._training_reservation_bytes,layout=learner.network.layout)


def test_fast_packed_phase_trims_global_event_padding_before_any_gather(monkeypatch):
    from gf_models.structured_learning import pack_window
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_update_planning import execution_plan
    tiny=learner_for('gru')
    model=replace(tiny.model,event=replace(tiny.model.event,max_interval=4096,max_episode=8192))
    learner=HpoLearner(model,replace(HPOTrainingConfig(),epochs=1),17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    source,table=rollout_for(learner,count=1).endpoint(0)
    base=source[0]
    frames=[]
    end=0
    for index in range(33):
        count=4096 if index==0 else 1
        frame=replace(base,revision=index,event_start=end,event_end=end+count,
            interval_numerics=(base.interval_numerics[0],)*count,
            interval_present=(base.interval_present[0],)*count,
            interval_categories=(base.interval_categories[0],)*count)
        validate_frame_tokens(frame,layout=learner.network.layout)
        frames.append(frame)
        end+=count
    validate_candidate_tokens(table,frames[-1],model.max_positions,layout=learner.network.layout)
    plan=execution_plan([(tuple(frames),table)],model,learner.binding,32,layout=learner.network.layout,
        reservation_bytes=learner._training_reservation_bytes,resident_bytes=0)
    assert plan.staging is None  # Exercise the admitted fast path, not avoidance.
    packed=pack_window(learner.network,frames,[table],[(0,-.5,0.,1.,.1)])
    observed=[]
    def inspect(tokens,raw,*,batch_lengths):
        for name in ('interval_numerics','interval_present','interval_categories'):
            actual=tuple(raw(name).shape)
            assert actual[:2]==(len(batch_lengths),max(batch_lengths))
            observed.append(actual)
        # This is a pure shape check: never allocate the old212/322MiB tensors.
        extent=len(batch_lengths)*max(batch_lengths)
        assert extent*learner.network.layout.event_width*4<=model.resources.max_tensor_bytes
        assert extent*max(len(v)+2 for v in learner.network.layout.event_categories)*8<=model.resources.max_tensor_bytes
        return torch.zeros(len(batch_lengths),model.temporal.hidden_size),torch.zeros(len(batch_lengths),18,1)
    monkeypatch.setattr(learner.network,'_encode_frame_operands',inspect)
    learner.network._learning_frames(packed,[0])
    learner.network._learning_frames(packed,list(range(1,33)))
    assert observed[0][:2]==(1,4096) and observed[-1][:2]==(32,1)


def test_default32_refines_suffix_temporaries_without_dropping_retained_graphs():
    from math import prod

    from gf_models.structured_learning import frame_operand_shapes
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_update_planning import execution_plan
    tiny=learner_for('gru')
    model=replace(tiny.model,hand=replace(tiny.model.hand,attention_feedforward=512,attention_layers=2))
    learner=HpoLearner(model,replace(HPOTrainingConfig(),epochs=1),17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    endpoint=controlled_endpoint(learner)
    resident=256*1024**2
    plan=execution_plan([endpoint],model,learner.binding,32,layout=learner.network.layout,
        reservation_bytes=learner._training_reservation_bytes,resident_bytes=resident)
    assert plan.staging is not None and len(plan.staging.suffix_chunks)>1
    assert sorted(index for chunk in plan.staging.suffix_chunks for index in chunk)==list(range(992,1024))
    retained=sum(sum(prod(shape)*width for shape,width in frame_operand_shapes(
        [endpoint[0][index] for index in chunk],learner.network.layout).values())
        for chunk in plan.staging.suffix_chunks)
    assert plan.estimate['retained_input_bytes']>=retained
    assert plan.estimate['total_bytes']+resident<=learner._training_reservation_bytes


def test_resolved_route_fact_is_successful_actual_branch_and_clears_on_failure(monkeypatch):
    learner=learner_for(recipe=replace(HPOTrainingConfig(),epochs=1))
    rollout=rollout_for(learner,count=3)
    learner._update_numeric(rollout,{},0.,learner.network)
    assert learner._last_update_execution=='microbatch'
    def fail(*args,**kwargs):
        raise ContractError('actual numerical failure')
    monkeypatch.setattr(learner,'_update_experimental',fail)
    with pytest.raises(ContractError,match='actual numerical'):
        learner._update_numeric(rollout,{},0.,learner.network)
    assert learner._last_update_execution is None


def test_extreme_rejection_precedes_shuffle_gradient_and_operand_mutation(monkeypatch):
    import copy

    from gf_common.domain import canonical
    from gf_models import structured_learning
    from gf_training import hpo_update_execution
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_update_planning import UpdateCapacityError
    tiny=learner_for('gru')
    model=replace(tiny.model,hand=replace(tiny.model.hand,attention_feedforward=1024,attention_heads=1,attention_layers=1))
    learner=HpoLearner(model,replace(HPOTrainingConfig(),epochs=1,tbptt_length=1024),17,tiny.binding,RESUME,actions=ACTIONS,rules=RULES)
    endpoint=controlled_endpoint(learner)
    class Input:
        resident_bytes=0
        used_decisions=collected_decisions=1
        def __len__(self):
            return 1
        def endpoint(self,index):
            return endpoint
        def vector(self,index):
            pytest.fail('loss operands before capacity preflight')
    def forbidden(*args,**kwargs):
        pytest.fail('operand allocation before explicit capacity refusal')
    monkeypatch.setattr(hpo_update_execution,'pack_window',forbidden)
    monkeypatch.setattr(hpo_update_execution,'pack_candidates',forbidden)
    monkeypatch.setattr(structured_learning,'pack_frames',forbidden)
    before=copy.deepcopy(learner.shuffle.bit_generator.state)
    weights={name:value.clone() for name,value in learner.network.state_dict().items()}
    with pytest.raises(UpdateCapacityError,match='capacity|reservation|retained'):
        learner._update_numeric(Input(),{},0.,learner.network)
    assert canonical(before)==canonical(learner.shuffle.bit_generator.state)
    assert learner.steps==0 and not learner.optimizer.state
    assert all(parameter.grad is None for parameter in learner.network.parameters())
    assert all(torch.equal(value,weights[name]) for name,value in learner.network.state_dict().items())
