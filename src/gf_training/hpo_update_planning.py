"""Pure endpoint grouping and joint preallocation admission for update experiments."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from gf_common.domain import ContractError
from gf_models.structured_config import dimensions, resource_estimate
from gf_models.structured_learning import operand_shapes


@dataclass(frozen=True)
class UpdateGroup:
    indices: tuple[int, ...]
    shared: bool = False


def frame_identity(frame: Any) -> Any:
    if hasattr(frame, '_banks'):
        return (id(frame._banks),tuple(sorted(frame._keys.items())),tuple(sorted(frame._scalars.items())))
    return id(frame)


def group_frames(endpoints: list[Any], model: Any, shared: bool) -> list[Any]:
    if shared:
        longest: Any = endpoints[0][0]
        for history,_ in endpoints:
            if len(history)>len(longest):
                longest=history
        return list(longest)
    return [frame for frames,_ in endpoints for frame in (frames[-1:] if model.model_id=='mlp' else frames)]


def joint_estimate(endpoints: list[Any], model: Any, binding: Any, tbptt: int, *,
                   layout: Any, shared: bool = False) -> dict[str, Any]:
    """One conservative formula for planning, learner and actual-phase admission.

    Includes the maximum of legacy retained endpoints and aggregate effective
    shape activations. Packed inputs include construction + CPU + GPU coexistence.
    Every effective tensor has its own independent limit before any allocation.
    """
    if not endpoints:
        raise ContractError('nonempty experimental group required')
    batch = len(endpoints)
    if batch>16:
        raise ContractError('experimental group exceeds 16 endpoints')
    frames = group_frames(endpoints,model,shared)
    candidates = [tokens for _,tokens in endpoints]
    width = max(len(c.kinds) for c in candidates)
    max_length = max(len(f) for f,_ in endpoints)
    total_frames = len(frames)
    d = dimensions(model,layout=layout)
    legacy = [resource_estimate(model,binding,decisions=len(f),
        interval_events=max(len(row.interval_numerics) for row in f),candidates=width,
        suffix_decisions=min(len(f),tbptt),layout=layout) for f,_ in endpoints]
    fixed = legacy[0]['parameter_optimizer_cpu_reference_bytes']+legacy[0]['context_workspace_bytes']
    shapes=operand_shapes(frames,candidates,model,layout)
    packed = sum(prod(shape)*size for shape,size in shapes.values())
    tensors = {name: prod(shape)*size for name,(shape,size) in shapes.items()}
    for size in (1,4,8):
        tensors[f'packed_dtype_slab_{size}'] = sum(prod(shape)*n for shape,n in shapes.values() if n==size)
    # Aggregate prefix and suffix event/frame encoders are simultaneously live;
    # never reuse the legacy serial-prefix/suffix-only event multiplier.
    events = max(len(f.interval_numerics) for f in frames)
    hidden = model.temporal.hidden_size
    actor = model.heads.actor_hidden_sizes[0]
    chunk = min(width,binding.candidate_chunk)
    tensors.update(
        frame_state=total_frames*layout.state_width*4,
        frame_card_input=total_frames*18*d['card_input']*4,
        frame_cards=total_frames*18*d['card_actor']*4,
        frame_hand=total_frames*18*max(model.card.output_size,model.hand.latent_size)*4,
        frame_hand_scores=total_frames*model.hand.attention_heads*18*18*4,
        frame_event_input=total_frames*events*layout.event_width*4,
        frame_event_gates=total_frames*events*model.event.hidden_size*4*3,
        frame_event_hidden=total_frames*events*model.event.hidden_size*4,
        frame_output=total_frames*d['frame']*4,
        projected_frames=total_frames*hidden*4,
        temporal_scores=(1 if shared else batch)*model.temporal.num_heads*max_length*max_length*4,
        temporal_hidden=(1 if shared else batch)*max_length*hidden*4,
        candidate_features=batch*width*layout.candidate_width*4,
        candidate_gather=batch*chunk*model.max_positions*actor*4,
        candidate_projection=batch*model.max_positions*19*actor*4,
        candidate_chunk=batch*chunk*actor*4,
        candidate_logits=batch*width*4,
        candidate_dense_reference=batch*chunk*d['actor']*4)
    for index,width_hidden in enumerate(model.state.hidden_sizes):
        tensors[f'state_layer_{index}'] = total_frames*width_hidden*4
    for index,width_hidden in enumerate(model.hand.hidden_sizes):
        tensors[f'hand_layer_{index}'] = total_frames*18*width_hidden*4
    tensors['hand_direct'] = total_frames*18*layout.card_direct_width*4
    tensors['card_onehot'] = total_frames*18*sum(len(v)+2 for v in layout.card_categories)*4
    tensors['temporal_feedforward'] = (1 if shared else batch)*max_length*model.temporal.feedforward_size*4
    # one_hot first allocates INT64, then FP32, before concatenation.
    # Model-hidden/output dimensions and every real dtype count independently;
    # parameter or aggregate admission cannot prove these tensor caps.
    onehots: dict[str, int] = {}
    for index,vocabulary in enumerate(layout.state_categories):
        onehots[f'state_onehot_int64_{index}'] = total_frames*(len(vocabulary)+2)*8
    onehots['opponent_onehot_int64'] = total_frames*18*(len(layout.card_categories[0])+2)*8
    for index,vocabulary in enumerate(layout.card_categories):
        onehots[f'card_onehot_int64_{index}'] = total_frames*18*(len(vocabulary)+2)*8
    for index,vocabulary in enumerate(layout.event_categories):
        onehots[f'event_onehot_int64_{index}'] = total_frames*events*(len(vocabulary)+2)*8
    onehots['candidate_kind_onehot_int64'] = batch*width*(len(layout.action_kinds)+2)*8
    onehots['candidate_target_onehot_int64'] = batch*width*(len(layout.targets)+2)*8
    tensors.update(onehots)
    for name,size in onehots.items():
        tensors[name.replace('int64','float32')] = size//2
    tensors['hand_attention_feedforward'] = total_frames*18*model.hand.attention_feedforward*4
    tensors['definition_embedding_output'] = total_frames*18*model.card.definition_embedding*4
    tensors['ability_embedding_output'] = total_frames*18*model.card.ability_embedding*4
    for index,width_hidden in enumerate(model.heads.actor_hidden_sizes):
        tensors[f'actor_layer_{index}'] = batch*chunk*width_hidden*4
    for index,width_hidden in enumerate(model.heads.critic_hidden_sizes):
        tensors[f'critic_layer_{index}'] = batch*width_hidden*4
    if any(size>model.resources.max_tensor_bytes for size in tensors.values()):
        raise ContractError('experimental effective tensor exceeds configured 64MiB limit')
    # Include every MLP layer output, recurrent gate/history, attention QKV,
    # softmax/workspaces, checkpoint replay AND backward coexistence. Factors
    # below are copy envelopes; the required conservative factor4 is outside.
    frame_mlp = total_frames*4*(sum(model.state.hidden_sizes)+model.state.output_size
        +18*sum(model.hand.hidden_sizes)+18*model.hand.latent_size)
    frame_activation = (sum(tensors[name] for name in ('frame_state','frame_card_input','frame_cards',
        'frame_hand','frame_event_input','frame_output','projected_frames'))*4
        + tensors['frame_event_gates']*model.event.num_layers*12
        + tensors['frame_event_hidden']*model.event.num_layers*4
        + tensors['frame_hand_scores']*model.hand.attention_layers*8 + frame_mlp*8
        + (tensors['hand_direct']+tensors['card_onehot'])*4
        + tensors['hand_attention_feedforward']*model.hand.attention_layers*8
        + tensors['definition_embedding_output']*4+tensors['ability_embedding_output']*4
        + sum(size for name,size in tensors.items() if 'onehot' in name)*4)
    temporal_activation = tensors['temporal_hidden']*model.temporal.num_layers*24
    if model.model_id=='transformer':
        temporal_activation += (tensors['temporal_scores']*8*model.temporal.num_layers
            +(1 if shared else batch)*max_length*model.temporal.feedforward_size*4*8*model.temporal.num_layers)
    candidate_activation = (tensors['candidate_features']*4+tensors['candidate_gather']*4
        +tensors['candidate_projection']*4+tensors['candidate_logits']*8
        +batch*chunk*sum(model.heads.actor_hidden_sizes)*4*16
        +batch*sum(model.heads.critic_hidden_sizes)*4*8)
    activation = max(sum(e['activation_and_copy_bytes'] for e in legacy),
        4*(frame_activation+temporal_activation+candidate_activation))
    metadata = 65536+total_frames*512+batch*1024
    return {'fixed_bytes':fixed,'activation_bytes':activation,'packed_bytes':packed,
        'metadata_bytes':metadata,'total_bytes':fixed+activation+packed*4+metadata,
        'tensor_bytes':tensors,'frame_count':total_frames}


def admit_group(rollout: Any, indices: list[int] | tuple[int,...], model: Any, binding: Any,
                tbptt: int, *, layout: Any, reservation_bytes: int, shared: bool = False) -> dict[str, Any]:
    endpoints = [rollout.endpoint(index) for index in indices]
    estimate = joint_estimate(endpoints,model,binding,tbptt,layout=layout,shared=shared)
    cap = min(reservation_bytes,model.resources.max_owned_bytes,binding.managed_budget_bytes)
    if estimate['total_bytes']+rollout.resident_bytes>cap:
        raise ContractError('experimental joint inputs/activations exceed SAME learner reservation')
    return estimate


def plan_groups(rollout: Any, logical: list[int], model: Any, binding: Any, tbptt: int, *,
                variant: str, reservation_bytes: int, layout: Any) -> list[UpdateGroup]:
    if variant=='shared-prefix' and model.model_id=='mlp':
        raise ContractError('shared-prefix is sequence-only')
    endpoints={index:rollout.endpoint(index) for index in logical}
    bins: dict[Any,list[int]]={}
    shared_bins: dict[Any,list[int]]={}
    for index in logical:
        frames,_=endpoints[index]
        cut=max(0,len(frames)-tbptt)
        if variant=='shared-prefix' and cut==0:
            key=(frame_identity(frames[0]),frames[0].actor,frames[0].episode,cut)
            shared_bins.setdefault(key,[]).append(index)
        else:
            bins.setdefault((0,1) if model.model_id=='mlp' else (cut,min(len(frames),tbptt)),[]).append(index)
    accepted: list[UpdateGroup]=[]
    def emit(indices: list[int], shared: bool) -> None:
        current: list[int]=[]
        for index in sorted(indices,key=lambda i:len(endpoints[i][1].kinds)):
            trial=[*current,index]
            width=max(len(endpoints[i][1].kinds) for i in trial)
            fits=len(trial)<=16 and len(trial)*width<=2*sum(len(endpoints[i][1].kinds) for i in trial)
            if fits:
                try:
                    admit_group(rollout,trial,model,binding,tbptt,layout=layout,
                        reservation_bytes=reservation_bytes,shared=shared)
                except ContractError:
                    fits=False
            if not fits:
                if current:
                    accepted.append(UpdateGroup(tuple(current),shared and len(current)>1))
                current=[index]
                admit_group(rollout,current,model,binding,tbptt,layout=layout,
                    reservation_bytes=reservation_bytes,shared=False)
            else:
                current=trial
        if current:
            accepted.append(UpdateGroup(tuple(current),shared and len(current)>1))
    for indices in shared_bins.values():
        longest=max((endpoints[i][0] for i in indices),key=len)
        matching=all(all(frame_identity(frame)==frame_identity(longest[pos]) for pos,frame in enumerate(endpoints[i][0])) for i in indices)
        if len(indices)>=2 and matching:
            emit(indices,True)
        else:
            for index in indices:
                length=len(endpoints[index][0])
                bins.setdefault((0,length),[]).append(index)
    for indices in bins.values():
        emit(indices,False)
    return accepted
