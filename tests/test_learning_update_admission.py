"""Pure effective-shape regressions: oversized tensors never reach allocation."""
from dataclasses import replace

import pytest

from gf_common.domain import ContractError
from gf_training.hpo_update_planning import admit_group, plan_groups
from tests.test_hpo_numeric import learner_for, rollout_for


@pytest.mark.parametrize('kind,batch,length,ff,actor,width',[
    ('gru',16,100,1024,None,None),
    ('mlp',16,1,None,(1,1024),1100),
    ('gru',2,900,None,None,None),
])
def test_effective_hidden_and_int64_onehot_tensors_reject_and_split_before_packing(kind,batch,length,ff,actor,width):
    learner=learner_for(kind)
    frames,table=rollout_for(learner).endpoint(0)
    model,binding=learner.model,learner.binding
    if ff:
        model=replace(model,hand=replace(model.hand,attention_feedforward=ff))
    if actor:
        model=replace(model,max_positions=1,heads=replace(model.heads,actor_hidden_sizes=actor))
    if width:
        binding=replace(binding,max_candidates=width,candidate_chunk=width)
        table=replace(table,kinds=(table.kinds[0],)*width,targets=(table.targets[0],)*width,
            numerics=(table.numerics[0],)*width,present=(table.present[0],)*width,
            offsets=tuple(range(width+1)),slots=(table.slots[0],)*width)
    endpoints=[(tuple([frames[0]]*length),table)]*batch
    class Input:
        resident_bytes=0
        def endpoint(self,index):
            return endpoints[index]
    with pytest.raises(ContractError,match='effective tensor'):
        admit_group(Input(),list(range(batch)),model,binding,32,layout=learner.network.layout,
            reservation_bytes=binding.managed_budget_bytes)
    groups=plan_groups(Input(),list(range(batch)),model,binding,32,variant='microbatch',
        layout=learner.network.layout,reservation_bytes=binding.managed_budget_bytes)
    assert len(groups)>1
    assert sorted(index for group in groups for index in group.indices)==list(range(batch))
    for group in groups:
        size=admit_group(Input(),group.indices,model,binding,32,layout=learner.network.layout,
            reservation_bytes=binding.managed_budget_bytes)
        assert max(size['tensor_bytes'].values())<=model.resources.max_tensor_bytes


@pytest.mark.parametrize('count,tail',[(181,53),(133,5),(175,47),(135,7)])
def test_true_logical128_tails_and_microbatch_weight_sum(count,tail):
    learner=learner_for()
    endpoint=rollout_for(learner).endpoint(0)
    class Input:
        resident_bytes=0
        def endpoint(self,index):
            return endpoint
    order=list(reversed(range(count)))
    logical=[order[:128],order[128:]]
    assert [len(batch) for batch in logical]==[128,tail]
    for batch in logical:
        groups=plan_groups(Input(),batch,learner.model,learner.binding,32,variant='microbatch',
            layout=learner.network.layout,reservation_bytes=learner.binding.managed_budget_bytes)
        assert sorted(i for group in groups for i in group.indices)==sorted(batch)
        assert sum(len(group.indices)/len(batch) for group in groups)==pytest.approx(1.)
