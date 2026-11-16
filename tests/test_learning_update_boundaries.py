"""Autograd, causality, empty branches and original owned retirement boundaries."""
from dataclasses import replace

import pytest

torch=pytest.importorskip('torch')

from gf_common.domain import ContractError
from gf_models.structured_learning import pack_window
from gf_training.hpo_training_execution import TrainingExecutionSpec
from tests.test_hpo_numeric import learner_for, rollout_for


@pytest.mark.parametrize('kind',['gru','transformer'])
def test_batched_tbptt_over32_detaches_only_prefix(kind):
    learner=learner_for(kind)
    rollout=rollout_for(learner,count=35)
    frames,candidates=rollout.endpoint(34)
    window=pack_window(learner.network,list(frames),[candidates],[rollout.vector(34)])
    observed=[]
    def capture(module,args,output):
        observed.append((tuple(output.shape),output.requires_grad))
    handle=learner.network.frame_projection.register_forward_hook(capture)
    try:
        logs,values=learner.network._learning_forward(window,[35],32)
        (values.sum()+logs[0,0]).backward()
    finally:
        handle.remove()
    assert observed==[((3,learner.model.temporal.hidden_size),False),
        ((32,learner.model.temporal.hidden_size),True)]
    assert learner.network.frame_projection.weight.grad is not None


@pytest.mark.parametrize('kind',['gru','transformer'])
def test_shared_prefix_causality_and_endpoint_specific_cards(kind):
    learner=learner_for(kind)
    rollout=rollout_for(learner,count=4)
    frames,last=rollout.endpoint(3)
    # Give the earlier endpoint its own distinct referenced card, as a real
    # hand/learned-card update would. Both action candidates reference slots0/1.
    distinct=[list(row) for row in frames[1].card_numerics]
    distinct[0][0]+=100
    frames=(*frames[:1],replace(frames[1],card_numerics=tuple(tuple(row) for row in distinct)),*frames[2:])
    first=rollout.endpoint(1)[1]
    for frame in frames:
        from gf_common.structured_tokens import validate_candidate_tokens, validate_frame_tokens
        validate_frame_tokens(frame,layout=learner.network.layout)
    validate_candidate_tokens(first,frames[1],learner.model.max_positions,layout=learner.network.layout)
    validate_candidate_tokens(last,frames[-1],learner.model.max_positions,layout=learner.network.layout)
    window=pack_window(learner.network,list(frames),[first,last],[rollout.vector(1),rollout.vector(3)])
    logs,values=learner.network._learning_forward(window,[2,4],4,shared=True)
    expected_log,expected_value,_=learner.network(frames[:2],first)
    torch.testing.assert_close(logs[0,:len(first.kinds)],expected_log)
    torch.testing.assert_close(values[0],expected_value)
    context,own_cards=learner.network.encode_history(frames[:2])
    _,last_cards=learner.network.encode_frame(frames[-1])
    assert not torch.equal(own_cards,last_cards)
    wrong_log,_=learner.network.distribution(context,last_cards,first)
    assert not torch.allclose(expected_log,wrong_log)
    altered=list(frames)
    numbers=list(altered[-1].state_numerics)
    numbers[0]+=10
    altered[-1]=replace(altered[-1],state_numerics=tuple(numbers))
    other=pack_window(learner.network,altered,[first,last],[rollout.vector(1),rollout.vector(3)])
    other_logs,other_values=learner.network._learning_forward(other,[2,4],4,shared=True)
    torch.testing.assert_close(logs[0],other_logs[0],rtol=0,atol=0)
    torch.testing.assert_close(values[0],other_values[0],rtol=0,atol=0)


def test_mixed_empty_event_and_hand_batches_keep_zero_branch_grad_none():
    from gf_common.structured_tokens import validate_candidate_tokens, validate_frame_tokens
    from gf_training.hpo_rollout import HpoRollout
    from tests.test_card_instance_learning import ACTIONS, RULES, records_for
    learner=learner_for()
    rollout=HpoRollout((records_for(learner,count=1,empty=True),),learner.model,
        learner.training,actions=ACTIONS,rules=RULES,selected_locations=((0,0),))
    frames,candidates=rollout.endpoint(0)
    current=frames[-1]
    assert not any(current.card_mask) and not candidates.slots
    empty=replace(current,event_start=current.event_end,
        interval_numerics=(),interval_present=(),interval_categories=())
    for frame in (empty,current):
        validate_frame_tokens(frame,layout=learner.network.layout)
        validate_candidate_tokens(candidates,frame,learner.model.max_positions,layout=learner.network.layout)
    vector=rollout.vector(0)
    for both_empty in (False,True):
        learner.optimizer.zero_grad(set_to_none=True)
        window=pack_window(learner.network,[empty,empty if both_empty else current],
            [candidates,candidates],[vector,vector])
        logs,values=learner.network._learning_forward(window,[1,1],1)
        assert torch.isfinite(logs).all() and torch.isfinite(values).all()
        (values.sum()+logs[:,0].sum()).backward()
        assert all((p.grad is None)==both_empty for p in learner.network.event_encoder.parameters())
        learner.optimizer.step()
        if both_empty:
            # The previous mixed group initialized state; it must retain its
            # actual step when the branch is unused on the next minibatch.
            assert all(learner.optimizer.state[p]['step']==1 for p in learner.network.event_encoder.parameters())


@pytest.mark.parametrize('variant',['metrics','packed','microbatch','shared-prefix'])
@pytest.mark.parametrize('failure',[False,True])
def test_owned_update_keeps_original_retirement_and_failure_poison(tmp_path,monkeypatch,variant,failure):
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_owned_numeric import captured_record
    learner=learner_for('gru')
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    numeric.persist(tmp_path/'numeric')
    learner._training_execution=TrainingExecutionSpec(variant=variant)
    learner._training_reservation_bytes=learner.binding.managed_budget_bytes
    if failure:
        real=numeric.vector
        def nonfinite(index):
            selected,old,value,advantage,_=real(index)
            return selected,old,value,advantage,float('inf')
        monkeypatch.setattr(numeric,'vector',nonfinite)
        # pytest retains exception tracebacks; clear their references before
        # checking original bank/FD retirement below.
        with pytest.raises(ContractError,match='endpoint loss') as caught:
            learner.update_owned(numeric)
        caught.value.__traceback__=None
        del caught
        assert numeric._state=='FAILED' and learner._owned_failed
        assert learner.steps==0 and not learner.optimizer.state
        with pytest.raises(ContractError,match='terminal'):
            learner.update_owned(numeric)
    else:
        result=learner.update_owned(numeric)
        assert result['endpoint_evaluations']==learner.training.epochs*3
        assert numeric._state=='UPDATED'
    numeric.close()
    numeric.require_retired()
    batch.close()
    snapshot.close()
