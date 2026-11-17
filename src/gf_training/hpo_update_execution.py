"""Private mechanisms used by the one HPO trainer's experimental loop."""
from __future__ import annotations

from typing import Any

from gf_common.domain import ContractError
from gf_common.hpo_owned_tokens import _OwnedFrameTokens, _require_owned_endpoint
from gf_models.structured_learning import StreamedLearningWindow, pack_candidates, pack_window
from gf_training.hpo_training_execution import TrainingExecutionSpec
from gf_training.hpo_update_planning import UpdateGroup, execution_plan, group_frames, plan_groups

METRICS = ('policy_loss','value_mse','entropy','approx_kl','total_loss')


def endpoint_terms(logprobs: Any, values: Any, mask: Any, selected: Any,
                   old: Any, advantage: Any, target: Any, *, training: Any) -> tuple[Any,Any]:
    import torch

    from gf_training.hpo_numeric import hpo_losses
    loss,measured = hpo_losses(logprobs,values,mask,selected,old,advantage,target,
        training=training,_endpoint_terms=True)
    if not torch.isfinite(loss).all():
        raise ContractError('nonfinite HPO PPO endpoint loss')
    return loss,measured


def execution_groups(learner: Any, rollout: Any, logical: list[int], spec: TrainingExecutionSpec) -> list[UpdateGroup]:
    if spec.variant in ('metrics','packed'):
        return [UpdateGroup((index,)) for index in logical]
    return plan_groups(rollout,logical,learner.model,learner.binding,learner.training.tbptt_length,
        variant=spec.variant,reservation_bytes=learner._training_reservation_bytes,layout=learner.network.layout)


def group_forward(learner: Any, rollout: Any, group: UpdateGroup, spec: TrainingExecutionSpec,
                  forward: Any) -> tuple[Any, Any, Any, Any, int, int]:
    torch = learner.torch
    endpoints = [rollout.endpoint(index) for index in group.indices]
    for frames,candidates in endpoints:
        if type(frames[0]) is _OwnedFrameTokens:
            _require_owned_endpoint(frames,candidates)
    vectors = [rollout.vector(index) for index in group.indices]
    lengths=[len(frames) for frames,_ in endpoints]
    suffix = learner._suffix_length(lengths[0])
    if spec.variant=='metrics':
        frames,candidates = endpoints[0]
        logs,value,_ = forward(frames,candidates,suffix_length=suffix)
        mask = torch.ones((1,len(candidates.kinds)),dtype=torch.bool,device=learner.binding.device)
        selected,old,_,advantage,target=vectors[0]
        selected_t = torch.tensor([selected],dtype=torch.int64,device=learner.binding.device)
        targets=[torch.tensor([number],dtype=torch.float32,device=learner.binding.device) for number in (old,advantage,target)]
        loss,metrics=endpoint_terms(logs[None],value[None],mask,selected_t,*targets,training=learner.training)
        return loss,metrics,None,None,(1 if learner.model.model_id=='mlp' else lengths[0]),4
    plan=group.plan or execution_plan(endpoints,learner.model,learner.binding,learner.training.tbptt_length,
        layout=learner.network.layout,reservation_bytes=learner._training_reservation_bytes,
        resident_bytes=rollout.resident_bytes,shared=group.shared)
    frames=group_frames(endpoints,learner.model,group.shared)
    window: Any
    if plan.staging is not None and spec.variant!='packed':
        candidates=pack_candidates(learner.network,[tokens for _,tokens in endpoints],vectors)
        window=StreamedLearningWindow(candidates,tuple(frames),plan.staging,[],len(frames),candidates.transfer_count)
        logs,values=learner.network._learning_streamed(window,lengths,suffix,shared=group.shared)
    else:
        if plan.staging is not None:
            from gf_training.hpo_update_planning import admit_group
            admit_group(rollout,group.indices,learner.model,learner.binding,learner.training.tbptt_length,
                layout=learner.network.layout,reservation_bytes=learner._training_reservation_bytes,shared=group.shared)
        window=pack_window(learner.network,frames,[tokens for _,tokens in endpoints],vectors)
        if spec.variant=='packed':
            logs,values=learner.network._learning_serial(window,*endpoints[0],suffix)
        else:
            logs,values=learner.network._learning_forward(window,lengths,suffix,shared=group.shared)
    raw=window.operands
    loss,metrics=endpoint_terms(logs,values,raw['mask'],raw['selected'],*raw['vectors'],training=learner.training)
    return loss,metrics,window,None,len(frames),window.transfer_count
