"""Immutable training operands, independent of trainer/Source and external I/O."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from gf_common.domain import ContractError


class LearningCapacityError(ContractError):
    """Pure effective-shape/residency refusal, never an execution/OOM retry."""

FRAME_FIELDS = ('state_numerics', 'state_present', 'state_categories', 'state_category_present',
    'opponent_learned', 'opponent_mask', 'card_categories', 'card_category_suffix',
    'card_numerics', 'card_present', 'card_mask', 'card_order', 'history',
    'interval_numerics', 'interval_present', 'interval_categories')
LONG_FIELDS = frozenset(('state_categories','opponent_learned','card_categories',
    'card_category_suffix','interval_categories','kinds','targets','references','selected'))
BOOL_FIELDS = frozenset(('card_mask','mask'))


def shape_of(value: Any) -> tuple[int, ...]:
    if hasattr(value, 'shape'):
        return tuple(value.shape)
    return (len(value), *shape_of(value[0])) if isinstance(value, (tuple,list)) and len(value) else (0,) if isinstance(value,(tuple,list)) else ()


def frame_shape(frame: Any, name: str, layout: Any, events: int) -> tuple[int, ...]:
    if name == 'opponent_mask':
        return shape_of(frame.opponent_learned)
    if name == 'card_category_suffix':
        return (18, len(layout.card_categories)-2)
    if name == 'card_order':
        return (18,)
    if name.startswith('interval_'):
        return (events, len(layout.event_categories) if name=='interval_categories' else len(layout.event_numeric_fields))
    return shape_of(getattr(frame,name))


def operand_shapes(frames: list[Any], candidates: list[Any], model: Any, layout: Any) -> dict[str, tuple[tuple[int,...], int]]:
    return {**frame_operand_shapes(frames,layout),**candidate_operand_shapes(candidates,model)}


@dataclass(frozen=True)
class PackedLearningWindow:
    operands: dict[str, Any]
    storage: tuple[Any, ...]
    cpu_storage: tuple[Any, ...]
    event_lengths: tuple[int, ...]
    frame_count: int
    transfer_count: int


def pack_window(network: Any, frames: list[Any], candidates: list[Any], vectors: list[Any]) -> PackedLearningWindow:
    """Caller admits ALL shapes/residency and validates endpoint liveness first.

    Each dtype crosses the device boundary once. No storage is reused between
    windows: checkpoint closures keep their tensor views until final backward.
    """
    return _pack(network,frames,candidates,vectors,
        operand_shapes(frames,candidates,network.config,network.layout),split=False)


def _pack(network: Any,frames: list[Any],candidates: list[Any],vectors: list[Any],
          shapes: dict[str,tuple[tuple[int,...],int]],*,split: bool) -> PackedLearningWindow:
    import torch
    dtype_for: dict[int,Any] = {8: np.dtype('int64'),4: np.dtype('float32'),1: np.dtype('bool')}
    segments=slab_plan(shapes,network.config.resources.max_tensor_bytes,split=split)
    slabs: list[Any]=[]
    arrays={}
    locations={}
    for segment in segments:
        size=shapes[segment[0]][1]
        slab=np.zeros(sum(prod(shapes[name][0]) for name in segment),dtype=dtype_for[size])
        offset=0
        for name in segment:
            shape=shapes[name][0]
            arrays[name]=slab[offset:offset+prod(shape)].reshape(shape)
            locations[name]=(len(slabs),offset,shape)
            offset+=prod(shape)
        slabs.append(slab)
    for index,frame in enumerate(frames):
        for name in FRAME_FIELDS:
            value = network._frame_operand_values(frame,name)
            if name.startswith('interval_'):
                if len(value):
                    arrays[name][index,:len(value)] = value
            else:
                arrays[name][index] = value
    for index,tokens in enumerate(candidates):
        width = len(tokens.kinds)
        for name in ('kinds','targets','numerics','present'):
            arrays[name][index,:width] = getattr(tokens,name)
        arrays['references'][index].fill(18)
        for row in range(width):
            slots = tokens.slots[tokens.offsets[row]:tokens.offsets[row+1]]
            arrays['references'][index,row,:len(slots)] = slots
        arrays['mask'][index,:width] = True
        selected,old,_,advantage,target = vectors[index]
        arrays['selected'][index] = selected
        arrays['vectors'][:,index] = old,advantage,target
    if frames:
        arrays['event_lengths'][:] = [len(f.interval_numerics) for f in frames]
    cpu=tuple(torch.from_numpy(slab) for slab in slabs)
    device=tuple(value.to(network.device) for value in cpu)
    operands={name:device[index][offset:offset+prod(shape)].view(shape)
        for name,(index,offset,shape) in locations.items()}
    # NumPy owners are deliberately frozen only after construction completes.
    for slab in slabs:
        slab.flags.writeable = False
    return PackedLearningWindow(operands,device,cpu,
        tuple(len(f.interval_numerics) for f in frames),len(frames),len(device))


@dataclass(frozen=True)
class FrameStagingPlan:
    prefix_chunks: tuple[tuple[int, ...], ...]
    suffix_chunks: tuple[tuple[int, ...], ...]


@dataclass
class StreamedLearningWindow:
    """Candidate storage plus immutable retained grad windows; prefix is transient."""
    candidates: PackedLearningWindow
    frames: tuple[Any, ...]
    staging: FrameStagingPlan
    retained: list[PackedLearningWindow]
    frame_count: int
    transfer_count: int

    @property
    def operands(self) -> dict[str,Any]:
        return self.candidates.operands


def slab_plan(shapes: dict[str,tuple[tuple[int,...],int]],limit: int,*,split: bool) -> tuple[tuple[str,...],...]:
    groups=[]
    for size in (8,4,1):
        current: list[str]=[]
        used=0
        for name,(shape,width) in shapes.items():
            if width!=size:
                continue
            count=prod(shape)*width
            if count>limit:
                raise LearningCapacityError('individual raw operand exceeds tensor capacity')
            if split and current and used+count>limit:
                groups.append(tuple(current))
                current=[]
                used=0
            current.append(name)
            used+=count
        if current:
            if used>limit:
                raise LearningCapacityError('aggregate raw dtype slab exceeds tensor capacity')
            groups.append(tuple(current))
    return tuple(groups)


def frame_operand_shapes(frames: list[Any],layout: Any) -> dict[str,tuple[tuple[int,...],int]]:
    events=max(len(frame.interval_numerics) for frame in frames)
    return {name:((len(frames),*frame_shape(frames[0],name,layout,events)),
        8 if name in LONG_FIELDS else 1 if name in BOOL_FIELDS else 4) for name in FRAME_FIELDS} | {
            'event_lengths':((len(frames),),8)}


def candidate_operand_shapes(candidates: list[Any],model: Any) -> dict[str,tuple[tuple[int,...],int]]:
    width=max(len(tokens.kinds) for tokens in candidates)
    return {name:((len(candidates),width,*shape_of(getattr(candidates[0],name))[1:]),
        8 if name in LONG_FIELDS else 4) for name in ('kinds','targets','numerics','present')} | {
        'references':((len(candidates),width,model.max_positions),8),
        'mask':((len(candidates),width),1),'selected':((len(candidates),),8),
        'vectors':((3,len(candidates)),4)}


def pack_frames(network: Any,frames: list[Any]) -> PackedLearningWindow:
    return _pack(network,frames,[],[],frame_operand_shapes(frames,network.layout),split=True)


def pack_candidates(network: Any,candidates: list[Any],vectors: list[Any]) -> PackedLearningWindow:
    return _pack(network,[],candidates,vectors,candidate_operand_shapes(candidates,network.config),split=True)
