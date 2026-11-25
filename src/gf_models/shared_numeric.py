"""Own-row numeric encoding and MLP math, independent of serving and training."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from gf_common.domain import ContractError, Json, closed
from gf_common.structured_tokens import validate_candidate_tokens, validate_frame_tokens
from gf_models.structured_config import recorded_token_signature
from gf_models.structured_learning import BOOL_FIELDS, FRAME_FIELDS, LONG_FIELDS


def numeric_row(frame: Any,candidates: Any,model: Any,*,layout: Any) -> Json:
    validate_frame_tokens(frame,layout=layout)
    validate_candidate_tokens(candidates,frame,model.max_positions,layout=layout)
    if model.model_id != 'mlp':
        raise ContractError('shared numeric API admits only MLP')
    fields = {name:getattr(frame,name) for name in FRAME_FIELDS
        if name not in ('opponent_mask','card_category_suffix','card_order')}
    fields.update(opponent_mask=tuple(value != 0 for value in frame.opponent_learned),
        card_category_suffix=tuple(row[2:] for row in frame.card_categories),
        card_order=tuple(row[layout.card_numeric_fields.index(layout.order_field)] for row in frame.card_numerics))
    fields.update({name:getattr(candidates,name) for name in ('kinds','targets','numerics','present')})
    fields['references'] = [list(candidates.slots[start:stop])+[18]*(model.max_positions-stop+start)
        for start,stop in zip(candidates.offsets[:-1],candidates.offsets[1:],strict=True)]
    shapes = {name:_shape(value) for name,value in fields.items()}
    # Empty intervals retain the exact trailing numerical/category widths.
    for name in ('interval_numerics','interval_present','interval_categories'):
        shapes[name] = [len(frame.interval_numerics),len(layout.event_categories) if name=='interval_categories' else len(layout.event_numeric_fields)]
    return {'fields':fields,'shapes':shapes}


def _shape(value: Any) -> list[int]:
    return [len(value),*_shape(value[0])] if isinstance(value,(list,tuple)) and value else [0] if isinstance(value,(list,tuple)) else []


def tensor_row(value: Json) -> Any:
    import torch
    from tensordict import TensorDict
    raw = closed(value,{'fields','shapes'})
    expected = set(FRAME_FIELDS)|{'kinds','targets','numerics','present','references'}
    closed(raw['fields'],expected)
    closed(raw['shapes'],expected)
    return TensorDict({name:torch.tensor(values,dtype=torch.long if name in LONG_FIELDS else
        torch.bool if name in BOOL_FIELDS else torch.float32).reshape(raw['shapes'][name])
        for name,values in raw['fields'].items()},batch_size=[])


def score_numeric_row(network: Any,row: Any) -> tuple[Any,Any]:
    """Every operation uses this row's own geometry and original reduction order."""
    if network.config.model_id != 'mlp':
        raise ContractError('shared scoring requires MLP')
    context, cards = encode_numeric_frame(network, row)
    return score_numeric_heads(network, row, context, cards)


def encode_numeric_frame(network: Any, row: Any) -> tuple[Any, Any]:
    tokens = SimpleNamespace(interval_numerics=range(row['interval_numerics'].shape[0]))
    return network._encode_frame_operands(tokens, lambda name: row[name])


def score_numeric_heads(network: Any, row: Any, context: Any, cards: Any) -> tuple[Any, Any]:
    import torch

    candidates = SimpleNamespace(kinds=range(row['kinds'].shape[0]),
        recipe_signature=recorded_token_signature(network.config,layout=network.layout))
    operands = {name:row[name][None] for name in ('kinds','targets','numerics','present','references')}
    scores = network.factorized_logits(context,cards,candidates,checkpoint=False,_operands=operands)
    logs,value = scores.log_softmax(-1),network.value(context)
    if not torch.isfinite(logs).all() or not torch.isfinite(value):
        raise ContractError('nonfinite native actor or critic output')
    return logs,value
