"""Immutable training operands, independent of trainer/Source and external I/O."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

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
    events = max(len(f.interval_numerics) for f in frames)
    count, width = len(frames), max(len(c.kinds) for c in candidates)
    shapes = {name: ((count, *frame_shape(frames[0], name, layout, events)),
        8 if name in LONG_FIELDS else 1 if name in BOOL_FIELDS else 4) for name in FRAME_FIELDS}
    shapes.update({name: ((len(candidates),width,*shape_of(getattr(candidates[0],name))[1:]),
        8 if name in LONG_FIELDS else 4) for name in ('kinds','targets','numerics','present')})
    shapes.update(references=((len(candidates),width,model.max_positions),8),
        mask=((len(candidates),width),1),selected=((len(candidates),),8),
        vectors=((3,len(candidates)),4),event_lengths=((count,),8))
    return shapes


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
    import torch
    shapes = operand_shapes(frames,candidates,network.config,network.layout)
    dtype_for = {8: np.dtype('int64'),4: np.dtype('float32'),1: np.dtype('bool')}
    slabs = {size: np.zeros(sum(prod(shape) for shape,n in shapes.values() if n==size),dtype=dtype)
        for size,dtype in dtype_for.items()}
    offsets = dict.fromkeys(slabs,0)
    arrays = {}
    for name,(shape,size) in shapes.items():
        offset = offsets[size]
        arrays[name] = slabs[size][offset:offset+prod(shape)].reshape(shape)
        offsets[size] += prod(shape)
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
    arrays['event_lengths'][:] = [len(f.interval_numerics) for f in frames]
    cpu = {size: torch.from_numpy(slab) for size,slab in slabs.items()}
    device = {size: value.to(network.device) for size,value in cpu.items()}
    operands = {}
    offsets = dict.fromkeys(slabs,0)
    for name,(shape,size) in shapes.items():
        offset=offsets[size]
        operands[name] = device[size][offset:offset+prod(shape)].view(shape)
        offsets[size] += prod(shape)
    # NumPy owners are deliberately frozen only after construction completes.
    for slab in slabs.values():
        slab.flags.writeable = False
    return PackedLearningWindow(operands,tuple(device.values()),tuple(cpu.values()),
        tuple(len(f.interval_numerics) for f in frames),len(frames),len(device))
