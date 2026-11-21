"""Private mechanisms used by the one HPO trainer's experimental loop."""
from __future__ import annotations

from typing import Any

from gf_common.domain import ContractError, Json
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


def execution_groups(learner: Any, rollout: Any, logical: list[int], spec: TrainingExecutionSpec, *,
                     _device_bytes: int | None = None) -> list[UpdateGroup]:
    from gf_training.rust_ppo_rollout import _RustTrainingRollout

    native = type(rollout) is _RustTrainingRollout
    if spec.variant in ('metrics','packed') or (native and learner.binding.device == 'cpu'):
        return [UpdateGroup((index,)) for index in logical]
    native_options: Json = {'_native_pack_fits': rollout.pack_fits} if native else {}
    return plan_groups(rollout,logical,learner.model,learner.binding,learner.training.tbptt_length,
        variant=spec.variant,reservation_bytes=learner._training_reservation_bytes,layout=learner.network.layout,
        device_bytes=learner._training_device_bytes if _device_bytes is None else _device_bytes, **native_options)


def group_forward(learner: Any, rollout: Any, group: UpdateGroup, spec: TrainingExecutionSpec,
                  forward: Any, *, _device_bytes: int | None = None) -> tuple[Any, Any, Any, Any, int, int]:
    from gf_training.rust_ppo_rollout import _RustTrainingRollout

    torch = learner.torch
    native = type(rollout) is _RustTrainingRollout
    device_bytes = learner._training_device_bytes if _device_bytes is None else _device_bytes
    endpoints = [rollout.endpoint(index) for index in group.indices]
    for frames,candidates in endpoints:
        if type(frames[0]) is _OwnedFrameTokens:
            _require_owned_endpoint(frames,candidates)
    lengths=[len(frames) for frames,_ in endpoints]
    suffix = learner._suffix_length(lengths[0])
    if spec.variant=='metrics' and not native:
        frames,candidates = endpoints[0]
        logs,value,_ = forward(frames,candidates,suffix_length=suffix)
        mask = torch.ones((1,len(candidates.kinds)),dtype=torch.bool,device=learner.binding.device)
        selected,old,_,advantage,target=rollout.vector(group.indices[0])
        selected_t = torch.tensor([selected],dtype=torch.int64,device=learner.binding.device)
        targets=[torch.tensor([number],dtype=torch.float32,device=learner.binding.device) for number in (old,advantage,target)]
        loss,metrics=endpoint_terms(logs[None],value[None],mask,selected_t,*targets,training=learner.training)
        return loss,metrics,None,None,(1 if learner.model.model_id=='mlp' else lengths[0]),4
    plan=group.plan or execution_plan(endpoints,learner.model,learner.binding,learner.training.tbptt_length,
        layout=learner.network.layout,reservation_bytes=learner._training_reservation_bytes,
        resident_bytes=rollout.resident_bytes,shared=group.shared,device_bytes=device_bytes)
    if device_bytes is not None:
        from gf_training.hpo_update_planning import admit_execution_plan
        admit_execution_plan(plan,learner.model,learner.binding,reservation_bytes=learner._training_reservation_bytes,
            resident_bytes=rollout.resident_bytes,device_bytes=device_bytes)
    frames=group_frames(endpoints,learner.model,group.shared)
    window: Any
    if native:
        scalar = learner.binding.device == 'cpu' or spec.variant in ('metrics', 'packed')
        if plan.staging is not None and not scalar:
            candidates = rollout.pack(learner.network, group.indices, 'candidates')
            window = StreamedLearningWindow(candidates, tuple(frames), plan.staging, [], len(frames), candidates.transfer_count)
            def pack_native_frames(indices: tuple[int, ...]) -> Any:
                return rollout.pack(learner.network, tuple(group.indices[index] for index in indices), 'frames')
            logs, values = learner.network._learning_streamed(window, lengths, suffix, _pack_frames=pack_native_frames)
        else:
            if plan.staging is not None:
                from gf_training.hpo_update_planning import admit_group
                admit_group(rollout, group.indices, learner.model, learner.binding, learner.training.tbptt_length,
                    layout=learner.network.layout, reservation_bytes=learner._training_reservation_bytes,
                    device_bytes=device_bytes)
            window = rollout.pack(learner.network, group.indices)
            logs, values = (learner.network._learning_serial(window, *endpoints[0], suffix) if scalar else
                learner.network._learning_forward(window, lengths, suffix))
    elif plan.staging is not None and spec.variant!='packed':
        vectors = [rollout.vector(index) for index in group.indices]
        candidates=pack_candidates(learner.network,[tokens for _,tokens in endpoints],vectors)
        window=StreamedLearningWindow(candidates,tuple(frames),plan.staging,[],len(frames),candidates.transfer_count)
        logs,values=learner.network._learning_streamed(window,lengths,suffix,shared=group.shared)
    else:
        vectors = [rollout.vector(index) for index in group.indices]
        if plan.staging is not None:
            from gf_training.hpo_update_planning import admit_group
            admit_group(rollout,group.indices,learner.model,learner.binding,learner.training.tbptt_length,
                layout=learner.network.layout,reservation_bytes=learner._training_reservation_bytes,shared=group.shared,
                device_bytes=device_bytes)
        window=pack_window(learner.network,frames,[tokens for _,tokens in endpoints],vectors)
        if spec.variant=='packed':
            logs,values=learner.network._learning_serial(window,*endpoints[0],suffix)
        else:
            logs,values=learner.network._learning_forward(window,lengths,suffix,shared=group.shared)
    raw=window.operands
    loss,metrics=endpoint_terms(logs,values,raw['mask'],raw['selected'],*raw['vectors'],training=learner.training)
    return loss,metrics,window,None,len(frames),window.transfer_count
