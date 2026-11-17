"""Pure bounded grouping and lifetime-aware Model4 execution admission."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from gf_common.domain import ContractError
from gf_models.structured_config import dimensions, resource_estimate
from gf_models.structured_learning import (
    FrameStagingPlan,
    LearningCapacityError,
    candidate_operand_shapes,
    frame_operand_shapes,
    operand_shapes,
    slab_plan,
)

UpdateCapacityError=LearningCapacityError


@dataclass(frozen=True)
class ExecutionPlan:
    staging: FrameStagingPlan | None
    estimate: dict[str,Any]


@dataclass(frozen=True)
class UpdateGroup:
    indices: tuple[int,...]
    shared: bool=False
    plan: ExecutionPlan | None=None


def frame_identity(frame: Any) -> Any:
    if hasattr(frame,'_banks'):
        return (id(frame._banks),tuple(sorted(frame._keys.items())),tuple(sorted(frame._scalars.items())))
    return id(frame)


def group_frames(endpoints: list[Any],model: Any,shared: bool) -> list[Any]:
    if shared:
        longest: Any=endpoints[0][0]
        for history,_ in endpoints:
            if len(history)>len(longest):
                longest=history
        return list(longest)
    return [frame for frames,_ in endpoints for frame in (frames[-1:] if model.model_id=='mlp' else frames)]


def _check(tensors: dict[str,int],model: Any) -> None:
    if any(size>model.resources.max_tensor_bytes for size in tensors.values()):
        raise UpdateCapacityError('effective tensor exceeds configured64MiB capacity')


def _raw(shapes: Any,model: Any,*,split: bool) -> tuple[dict[str,int],int]:
    tensors={name:prod(shape)*size for name,(shape,size) in shapes.items()}
    _check(tensors,model)
    segments=slab_plan(shapes,model.resources.max_tensor_bytes,split=split)
    tensors.update({f'slab_{index}':sum(tensors[name] for name in segment) for index,segment in enumerate(segments)})
    return tensors,sum(prod(shape)*size for shape,size in shapes.values())


def _frame_geometry(frames: list[Any],model: Any,layout: Any) -> tuple[dict[str,int],int,int,int]:
    raw,packed=_raw(frame_operand_shapes(frames,layout),model,split=True)
    count=len(frames)
    events=max(len(frame.interval_numerics) for frame in frames)
    positive=sum(bool(len(frame.interval_numerics)) for frame in frames)
    d=dimensions(model,layout=layout)
    tensors=dict(raw)
    tensors.update(state_input=count*layout.state_width*4,card_input=count*18*d['card_input']*4,
        card_output=count*18*model.card.output_size*4,actor_cards=count*18*d['card_actor']*4,
        frame_features=count*d['frame']*4,projected=count*model.temporal.hidden_size*4,
        hand_latent=count*18*model.hand.latent_size*4,
        definition_embedding=count*18*model.card.definition_embedding*4,
        ability_embedding=count*18*model.card.ability_embedding*4,
        event_input=positive*events*layout.event_width*4,
        event_gates=positive*events*model.event.hidden_size*3*4,
        event_output=positive*events*model.event.hidden_size*4)
    for name,widths,multiplier in (('state',model.state.hidden_sizes,1),('hand',model.hand.hidden_sizes,18),
            ('dense_temporal',(model.temporal.hidden_size,)*(model.temporal.num_layers-1) if model.model_id=='mlp' else (),1)):
        for index,width in enumerate(widths):
            tensors[f'{name}_layer_{index}']=count*multiplier*width*4
    def onehot(name: str,size: int) -> None:
        tensors[name+'_int64']=size*8
        tensors[name+'_float32']=size*4
    for index,vocabulary in enumerate(layout.state_categories):
        onehot(f'state_onehot_{index}',count*(len(vocabulary)+2))
    onehot('opponent_onehot',count*18*(len(layout.card_categories[0])+2))
    for index,vocabulary in enumerate(layout.card_categories):
        if index>=2 or model.hand.kind=='direct':
            onehot(f'card_onehot_{index}',count*18*(len(vocabulary)+2))
    for index,vocabulary in enumerate(layout.event_categories):
        onehot(f'event_onehot_{index}',positive*events*(len(vocabulary)+2))
    hand_attention=0
    if model.hand.kind=='attention':
        tensors['hand_attention_scores']=count*model.hand.attention_heads*18*18*4
        tensors['hand_attention_feedforward']=count*18*model.hand.attention_feedforward*4
        hand_attention=model.hand.attention_layers*(2*tensors['hand_attention_scores']
            +2*tensors['hand_attention_feedforward']+8*tensors['card_output'])
    if model.hand.kind=='direct':
        tensors['hand_direct']=count*18*layout.card_direct_width*4
    _check(tensors,model)
    # Constant one-hot construction is temporary. Autograd retains concatenated
    # encoder inputs and actual differentiable layer/GRU/attention intermediates.
    saved=(tensors['state_input']+tensors['card_input']+2*tensors['card_output']
        +tensors['actor_cards']+tensors['frame_features']+2*tensors['projected']
        +tensors['hand_latent']+sum(size for name,size in tensors.items() if '_layer_' in name)
        +hand_attention+tensors['event_input']+tensors['event_output']*model.event.num_layers*16)
    return tensors,sum(tensors.values()),saved,packed


def _temporal_geometry(endpoints: list[Any],model: Any,tbptt: int,shared: bool) -> tuple[dict[str,int],int]:
    if model.model_id=='mlp':
        return {},0
    batch=1 if shared else len(endpoints)
    length=max(len(frames) for frames,_ in endpoints)
    suffix=length if shared else min(length,tbptt)
    prefix=length-suffix
    hidden=model.temporal.hidden_size
    tensors={'prefix_projected':batch*prefix*hidden*4,'suffix_projected':batch*suffix*hidden*4,
        'temporal_hidden':batch*length*hidden*4,'temporal_gates':batch*length*hidden*3*4}
    if model.model_id=='gru':
        envelope=model.temporal.num_layers*(tensors['prefix_projected']*8+tensors['suffix_projected']*20)
    else:
        heads=model.temporal.num_heads
        tensors.update(prefix_scores=batch*heads*prefix*prefix*4,
            suffix_scores=batch*heads*suffix*length*4,
            prefix_mask=batch*prefix*prefix,suffix_mask=batch*suffix*length,
            prefix_feedforward=batch*prefix*model.temporal.feedforward_size*4,
            suffix_feedforward=batch*suffix*model.temporal.feedforward_size*4,
            temporal_kv=batch*length*hidden*4)
        envelope=(2*tensors['prefix_scores']+2*tensors['prefix_feedforward']+tensors['prefix_projected']*8
            +model.temporal.num_layers*(2*tensors['suffix_scores']+2*tensors['suffix_feedforward']
                +tensors['suffix_projected']*8+4*tensors['temporal_kv']))
    _check(tensors,model)
    return tensors,envelope+2*tensors['prefix_projected']+2*tensors['suffix_projected']


def _candidate_geometry(candidates: list[Any],model: Any,binding: Any,layout: Any,*,split: bool) -> tuple[dict[str,int],int,int]:
    tensors,packed=_raw(candidate_operand_shapes(candidates,model),model,split=split)
    batch=len(candidates)
    width=max(len(tokens.kinds) for tokens in candidates)
    chunk=min(width,binding.candidate_chunk)
    first=model.heads.actor_hidden_sizes[0]
    tensors.update(candidate_features=batch*width*layout.candidate_width*4,
        candidate_gather=batch*chunk*model.max_positions*first*4,
        candidate_projection=batch*model.max_positions*19*first*4,
        global_logits=batch*width*4,
        candidate_kind_onehot_int64=batch*width*(len(layout.action_kinds)+2)*8,
        candidate_target_onehot_int64=batch*width*(len(layout.targets)+2)*8)
    for index,size in enumerate(model.heads.actor_hidden_sizes):
        tensors[f'actor_layer_{index}']=batch*chunk*size*4
    for index,size in enumerate(model.heads.critic_hidden_sizes):
        tensors[f'critic_layer_{index}']=batch*size*4
    _check(tensors,model)
    replay=sum(size for name,size in tensors.items() if name.startswith(('actor_layer_','critic_layer_')))
    envelope=(4*replay+2*tensors['candidate_gather']+2*tensors['candidate_projection']
        +2*tensors['candidate_features']+8*tensors['global_logits']
        +tensors['candidate_kind_onehot_int64']*2+tensors['candidate_target_onehot_int64']*2)
    return tensors,envelope,packed


def _estimate(endpoints: list[Any],model: Any,binding: Any,tbptt: int,*,layout: Any,
              shared: bool,staging: FrameStagingPlan | None) -> dict[str,Any]:
    if not endpoints or len(endpoints)>16:
        raise UpdateCapacityError('nonempty group of at most16 endpoints required')
    fixed=resource_estimate(model,binding,decisions=1,interval_events=0,candidates=1,suffix_decisions=1,layout=layout)
    fixed_bytes=fixed['parameter_optimizer_cpu_reference_bytes']+fixed['context_workspace_bytes']
    frames=group_frames(endpoints,model,shared)
    candidate=[tokens for _,tokens in endpoints]
    tensors,temporal=_temporal_geometry(endpoints,model,tbptt,shared)
    candidate_tensors,candidate_activation,candidate_packed=_candidate_geometry(candidate,model,binding,layout,split=staging is not None)
    tensors.update({'candidate.'+name:size for name,size in candidate_tensors.items()})
    prefix_indices=[]
    if model.model_id!='mlp' and not shared:
        length=len(endpoints[0][0])
        prefix=length-min(length,tbptt)
        prefix_indices=[row*length+pos for row in range(len(endpoints)) for pos in range(prefix)]
    prefix_set=set(prefix_indices)
    prefix_chunks: tuple[tuple[int,...],...]
    suffix_chunks: tuple[tuple[int,...],...]
    if staging is None:
        raw,packed=_raw(operand_shapes(frames,candidate,model,layout),model,split=False)
        tensors.update({'packed.'+name:size for name,size in raw.items()})
        prefix_chunks=(tuple(prefix_indices),) if prefix_indices else ()
        suffix_chunks=(tuple(index for index in range(len(frames)) if index not in prefix_set),)
        retained_packed=packed
    else:
        prefix_chunks,suffix_chunks=staging.prefix_chunks,staging.suffix_chunks
        retained_packed=candidate_packed
    prefix_peak=prefix_packed=temporary_peak=0
    retained_activation=0
    for gradient,chunks in ((False,prefix_chunks),(True,suffix_chunks)):
        for ordinal,indices in enumerate(chunks):
            geometry,temporary,saved,packed=_frame_geometry([frames[index] for index in indices],model,layout)
            tensors.update({f'{"grad" if gradient else "prefix"}.{ordinal}.{name}':size for name,size in geometry.items()})
            if gradient:
                retained_activation+=saved
                temporary_peak=max(temporary_peak,temporary)
                if staging is not None:
                    retained_packed+=packed
            else:
                prefix_peak=max(prefix_peak,temporary)
                if staging is not None:
                    prefix_packed=max(prefix_packed,packed)
    activation=4*(retained_activation+candidate_activation+temporal+max(prefix_peak,temporary_peak))
    packed_bytes=4*(retained_packed+prefix_packed)
    metadata=65536+len(frames)*512+len(endpoints)*1024
    return {'fixed_bytes':fixed_bytes,'activation_bytes':activation,'packed_bytes':packed_bytes,
        'retained_input_bytes':retained_packed,'prefix_temporary_bytes':prefix_packed,
        'retained_frame_activation_bytes':4*retained_activation,
        'total_bytes':fixed_bytes+activation+packed_bytes+metadata,'metadata_bytes':metadata,
        'tensor_bytes':tensors,'frame_count':len(frames),'streamed':staging is not None}


def joint_estimate(endpoints: list[Any],model: Any,binding: Any,tbptt: int,*,layout: Any,shared: bool=False) -> dict[str,Any]:
    return _estimate(endpoints,model,binding,tbptt,layout=layout,shared=shared,staging=None)


def _within(estimate: dict[str,Any],cap: int,resident: int) -> None:
    if estimate['total_bytes']+resident>cap:
        raise UpdateCapacityError('retained graph/input capacity exceeds SAME learner reservation')


def admit_group(rollout: Any,indices: Any,model: Any,binding: Any,tbptt: int,*,layout: Any,reservation_bytes: int,shared: bool=False) -> dict[str,Any]:
    estimate=joint_estimate([rollout.endpoint(index) for index in indices],model,binding,tbptt,layout=layout,shared=shared)
    _within(estimate,min(reservation_bytes,model.resources.max_owned_bytes,binding.managed_budget_bytes),rollout.resident_bytes)
    return estimate


def execution_plan(endpoints: list[Any],model: Any,binding: Any,tbptt: int,*,layout: Any,
                   reservation_bytes: int,resident_bytes: int,shared: bool=False) -> ExecutionPlan:
    cap=min(reservation_bytes,model.resources.max_owned_bytes,binding.managed_budget_bytes)
    try:
        estimate=joint_estimate(endpoints,model,binding,tbptt,layout=layout,shared=shared)
        _within(estimate,cap,resident_bytes)
        return ExecutionPlan(None,estimate)
    except UpdateCapacityError:
        pass  # Pure preallocation capacity refinement; no operation has run.
    frames=group_frames(endpoints,model,shared)
    length=len(endpoints[0][0])
    prefix=0 if shared or model.model_id=='mlp' else length-min(length,tbptt)
    prefixes={row*length+pos for row in range(len(endpoints)) for pos in range(prefix)}
    def chunks(indices: list[int],maximum: int) -> tuple[tuple[int,...],...]:
        bins: dict[int,list[int]]={}
        for index in indices:
            bins.setdefault(len(frames[index].interval_numerics),[]).append(index)
        groups=[]
        for rows in bins.values():
            current: list[int]=[]
            for index in rows:
                trial=[*current,index]
                fits=len(trial)<=maximum
                if fits:
                    try:
                        _frame_geometry([frames[i] for i in trial],model,layout)
                    except UpdateCapacityError:
                        fits=False
                if not fits and current:
                    groups.append(tuple(current))
                    current=[]
                _frame_geometry([frames[index]],model,layout)
                current.append(index)
            if current:
                groups.append(tuple(current))
        return tuple(groups)
    suffix_indices=[index for index in range(len(frames)) if index not in prefixes]
    maximum=64
    while True:
        # Finer grad chunks reduce only live construction/gather temporaries.
        # Every chunk's graph and input storage remains in the retained sum.
        staging=FrameStagingPlan(chunks(sorted(prefixes),maximum),chunks(suffix_indices,maximum))
        estimate=_estimate(endpoints,model,binding,tbptt,layout=layout,shared=shared,staging=staging)
        try:
            _within(estimate,cap,resident_bytes)
            return ExecutionPlan(staging,estimate)
        except UpdateCapacityError:
            if maximum==1:
                raise
            maximum=max(1,maximum//2)


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
    def planned(indices: list[int],shared: bool) -> UpdateGroup:
        plan=execution_plan([endpoints[i] for i in indices],model,binding,tbptt,layout=layout,
            reservation_bytes=reservation_bytes,resident_bytes=rollout.resident_bytes,shared=shared)
        return UpdateGroup(tuple(indices),shared,plan)
    def emit(indices: list[int], shared: bool) -> None:
        current: list[int]=[]
        for index in sorted(indices,key=lambda i:len(endpoints[i][1].kinds)):
            trial=[*current,index]
            width=max(len(endpoints[i][1].kinds) for i in trial)
            fits=len(trial)<=16 and len(trial)*width<=2*sum(len(endpoints[i][1].kinds) for i in trial)
            if fits:
                try:
                    execution_plan([endpoints[i] for i in trial],model,binding,tbptt,layout=layout,
                        reservation_bytes=reservation_bytes,resident_bytes=rollout.resident_bytes,shared=shared)
                except UpdateCapacityError:
                    fits=False
            if not fits:
                if current:
                    accepted.append(planned(current,shared and len(current)>1))
                current=[index]
                planned(current,False)
            else:
                current=trial
        if current:
            accepted.append(planned(current,shared and len(current)>1))
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
