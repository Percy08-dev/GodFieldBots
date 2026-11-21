"""Batch raw storage uses the actual numeric schema, including derived operands."""
from dataclasses import replace

import pytest


@pytest.mark.parametrize('version',[3,4])
def test_cuda_estimate_counts_complete_numeric_batch_and_both_returned_vectors(version):
    pytest.importorskip('torch')
    from gf_common.shared_inference import shared_runtime
    from gf_models.shared_admission import shared_service_estimate
    from gf_models.shared_numeric import numeric_row, tensor_row
    from gf_models.structured_config import InferenceBindingV2
    from tests.model3_support import LAYOUT, model3, raw_inputs
    from tests.test_hpo_network import small_model4
    model=model3() if version==3 else small_model4(depth=2)
    binding=InferenceBindingV2(device='cuda:0',max_candidates=64,candidate_chunk=2)
    frames,candidates,*_=raw_inputs(count=1)
    frame=frames[0]
    events=model.event.max_interval
    frame=replace(frame,event_end=events,interval_numerics=(frame.interval_numerics[0],)*events,
        interval_present=(frame.interval_present[0],)*events,interval_categories=(frame.interval_categories[0],)*events)
    count=binding.max_candidates
    candidates=replace(candidates,kinds=(candidates.kinds[0],)*count,targets=(candidates.targets[0],)*count,
        numerics=(candidates.numerics[0],)*count,present=(candidates.present[0],)*count,
        offsets=tuple(range(count+1)),slots=(candidates.slots[0],)*count)
    row=tensor_row(numeric_row(frame,candidates,model,layout=LAYOUT))
    actual=sum(tensor.numel()*tensor.element_size() for tensor in row.values())
    estimate=shared_service_estimate(model,binding,shared_runtime({'mode':'shared-mlp-v1','max_batch_size':4},game_concurrency=2),layout=LAYOUT)
    assert estimate['row_input_bytes']==actual
    assert estimate['batch_input_bytes']==4*actual
    assert estimate['retained_output_bytes']==4*(2*count*4+4+4*8)
