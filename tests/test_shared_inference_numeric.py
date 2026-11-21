"""Real shared MLP scoring uses isolated row geometry for private lengths."""
import os
from dataclasses import replace

import pytest

torch = pytest.importorskip('torch')

from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from tests.model3_support import LAYOUT, model3, raw_inputs
from tests.test_hpo_network import small_model4


@pytest.mark.parametrize('version',[3,4])
def test_numeric_row_matches_isolated_network_and_is_independent_of_private_rows(version):
    from gf_models.shared_numeric import numeric_row, score_numeric_row, tensor_row
    model = model3() if version == 3 else small_model4()
    binding = InferenceBindingV2(max_candidates=64,candidate_chunk=2)
    network = make_network(model,17,binding,layout=LAYOUT)
    frames,candidates,*_ = raw_inputs(count=3)
    row = tensor_row(numeric_row(frames[-1],candidates,model,layout=LAYOUT))
    other,candidate_other,*_ = raw_inputs(count=1,events=False,empty=True)
    changed = tensor_row(numeric_row(other[-1],candidate_other,model,layout=LAYOUT))
    with torch.no_grad():
        expected = network(frames,candidates)
        before = score_numeric_row(network,row)
        score_numeric_row(network,changed)
        after = score_numeric_row(network,row)
    torch.testing.assert_close(before[0],expected[0],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(before[1],expected[1],atol=1e-5,rtol=1e-5)
    assert torch.equal(before[0],after[0]) and torch.equal(before[1],after[1])
    a,b = torch.Generator().manual_seed(17),torch.Generator().manual_seed(17)
    assert torch.multinomial(before[0].exp(),1,generator=a).item() == torch.multinomial(after[0].exp(),1,generator=b).item()
    bad = replace(frames[-1],revision=frames[-1].revision+1)
    assert numeric_row(bad,candidates,model,layout=LAYOUT)['fields'] == numeric_row(frames[-1],candidates,model,layout=LAYOUT)['fields']


@pytest.mark.parametrize('version',[3,4])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_real_registry_batch_has_exact_private_value_length_order_and_trial_independence(version,device):
    assert_registry_private_independence(version,device,wrapped=False)


@pytest.mark.parametrize('version',[3,4])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_real_wrapped_registry_has_exact_private_value_length_order_and_trial_independence(version,device):
    assert_registry_private_independence(version,device,wrapped=True)


def assert_registry_private_independence(version,device,*,wrapped):
    if device!='cpu' and os.environ.get('GF_RUN_CUDA')!='1':pytest.skip('separate actual CUDA isolation check')
    from tensordict import NonTensorData, TensorDict, lazy_stack

    from gf_models.shared_numeric import numeric_row, tensor_row
    from gf_models.shared_registry import make_registry, publication
    model=model3() if version==3 else small_model4(depth=2)
    binding=InferenceBindingV2(device=device,max_candidates=64,candidate_chunk=2)
    network=make_network(model,17,binding,layout=LAYOUT)
    if wrapped:
        import multiprocessing as mp

        from gf_models.shared_telemetry import make_served_registry
        ctx=mp.get_context('spawn')
        block=ctx.Array('Q',8,lock=ctx.Lock())
        registry=make_served_registry(device=device,cache_entries=6,cache_weight_bytes=134217728,diagnostic_block=block)
    else:registry=make_registry(device=device,cache_entries=6,cache_weight_bytes=134217728)
    registry.load_policy_weights(publication(network,'a'*64))
    frames,candidates,*_=raw_inputs(count=3)
    reference=tensor_row(numeric_row(frames[-1],candidates,model,layout=LAYOUT))
    routing={'checkpoint':'a'*64,'execution':'e'*32,'game':'same','participant':1,'request':1,'revision':7}
    own=TensorDict({'inputs':reference,'routing':NonTensorData(routing)},batch_size=[])
    expected=registry(lazy_stack([own]).to(device)).unbind(0)[0]
    for empty,events in ((True,False),(False,True)):
        other,other_candidates,*_=raw_inputs(count=1,empty=empty,events=events)
        private=tensor_row(numeric_row(other[-1],other_candidates,model,layout=LAYOUT))
        private['history']=torch.full_like(private['history'],123456)
        for game,participant,execution in (('same',2,'e'*32),('other',1,'e'*32),('same',1,'f'*32)):
            peer=TensorDict({'inputs':private,'routing':NonTensorData(routing|{
                'game':game,'participant':participant,'execution':execution,'request':2})},batch_size=[])
            for order in ((own,peer),(peer,own)):
                result=registry(lazy_stack(list(order)).to(device)).unbind(0)[0 if order[0] is own else 1]
                assert torch.equal(expected['logs'],result['logs'])
                assert torch.equal(expected['probabilities'],result['probabilities'])
                assert torch.equal(expected['value'],result['value'])
                before,after=torch.Generator().manual_seed(17),torch.Generator().manual_seed(17)
                assert torch.multinomial(expected['probabilities'].cpu(),1,generator=before).item()==torch.multinomial(result['probabilities'].cpu(),1,generator=after).item()
