"""Deterministic FP32 Tree2 rows; current raw tokens only, no learned encoders."""
from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from gf_common.domain import ContractError, Json, closed, digest, integer
from gf_common.structured import ACTION_KINDS, TARGETS
from gf_common.structured_history import APPENDED_HISTORY_FIELDS
from gf_common.structured_tokens import (
    CARD_CATEGORIES,
    CARD_NUMERIC_FIELDS,
    EVENT_ACTORS,
    EVENT_NUMERIC_FIELDS,
    EVENT_TYPES,
    STATE_CATEGORIES,
    STATE_NUMERIC_FIELDS,
    CandidateTokens,
    FrameTokens,
    signature,
    validate_candidate_tokens,
    validate_frame_tokens,
)
from gf_models.structured_tree_config import (
    CARD_WIDTH,
    CONTEXT_WIDTH,
    TREE_NUMERICS,
    TREE_RECIPE,
    ConfiguredTreeModelV2,
    TreeDescriptor,
    TreeEncoderConfigV2,
    architecture_config,
    configured_tree_v2,
    feature_width,
    resource_estimate,
)

STATE_CATEGORY_FIELDS = ('phase', 'required_choice', 'first_player', 'active_player', 'current_player',
                         'result', 'own.disease', 'own.guardian', 'opponent.disease', 'opponent.guardian')
CARD_CATEGORY_FIELDS = ('definition', 'ability', 'kind', 'attribute', 'disease', 'guardian', 'source')
HISTORY_FIELDS = (tuple('event.' + name for name in EVENT_TYPES)
    + tuple('actor.' + field + '.' + role for field in EVENT_ACTORS for role in ('self', 'opponent'))
    + tuple('numeric.' + field + '.' + stat for field in EVENT_NUMERIC_FIELDS for stat in ('sum', 'present'))
    + APPENDED_HISTORY_FIELDS)


def categorical_limits(encoder: TreeEncoderConfigV2) -> tuple[int, ...]:
    return (tuple(len(values) + 1 for values in STATE_CATEGORIES)
        + tuple(len(values) + 1 for _ in range(18) for values in CARD_CATEGORIES)
        + (len(CARD_CATEGORIES[0]) + 1,) * 18 + (len(ACTION_KINDS) + 1, len(TARGETS) + 1)
        + tuple(len(values) + 1 for _ in range(encoder.max_positions) for values in CARD_CATEGORIES))


def _card_layout(prefix: str) -> tuple[tuple[str, str], ...]:
    return (tuple((prefix + '.numeric.' + name, 'numeric') for name in CARD_NUMERIC_FIELDS)
        + tuple((prefix + '.present.' + name, 'presence') for name in CARD_NUMERIC_FIELDS)
        + tuple((prefix + '.category.' + name, 'category') for name in CARD_CATEGORY_FIELDS)
        + ((prefix + '.mask', 'presence'),))


def feature_layout(encoder: TreeEncoderConfigV2) -> Json:
    if type(encoder) is not TreeEncoderConfigV2:
        raise ContractError('closed Tree2 encoder required')
    fields = (tuple(('state.numeric.' + name, 'numeric') for name in STATE_NUMERIC_FIELDS)
        + tuple(('state.present.' + name, 'presence') for name in STATE_NUMERIC_FIELDS)
        + tuple(('state.category.' + name, 'category') for name in STATE_CATEGORY_FIELDS)
        + tuple(('state.category_present.' + name, 'presence') for name in STATE_CATEGORY_FIELDS)
        + tuple(item for slot in range(18) for item in _card_layout(f'own.{slot}'))
        + tuple((f'opponent.learned.{slot}', 'category') for slot in range(18))
        + tuple(('history.' + name, 'numeric') for name in HISTORY_FIELDS)
        + (('candidate.kind', 'category'), ('candidate.target', 'category'))
        + tuple(('candidate.numeric.' + name, 'numeric') for name in ('hp', 'mp', 'money', 'accept'))
        + tuple(('candidate.present.' + name, 'presence') for name in ('hp', 'mp', 'money', 'accept'))
        + (('candidate.length', 'numeric'),)
        + tuple((f'candidate.used_slot.{slot}', 'numeric') for slot in range(18))
        + tuple(item for position in range(encoder.max_positions) for item in _card_layout(f'candidate.position.{position}')))
    if len(fields) != feature_width(encoder.max_positions) or len({name for name, _ in fields}) != len(fields):
        raise ContractError('Tree2 feature layout mismatch')
    layout = {'recipe': TREE_RECIPE, 'numeric_transform': TREE_NUMERICS, 'encoder': asdict(encoder),
              'token_signature': signature(encoder.aggregate_window), 'feature_names': [name for name, _ in fields],
              'categorical_indices': [i for i, (_, transform) in enumerate(fields) if transform == 'category'],
              'transforms': [transform for _, transform in fields]}
    return {**layout, 'feature_signature': digest(layout)}


def _owned(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        size += sum(_owned(item, seen) for item in value)
    elif hasattr(value, '__dataclass_fields__'):
        size += _owned(vars(value), seen)
    elif isinstance(value, dict):
        size += sum(_owned(key, seen) + _owned(item, seen) for key, item in value.items())
    return size


def signed_log_into(target: np.ndarray[Any, Any], values: Any) -> None:
    import numpy as np
    target[:] = values
    # All inputs are validated int64/closed categories; float32 conversion is
    # finite. Preserve NumPy's operation order with bounded/in-place ufuncs.
    signs = np.sign(target)
    np.abs(target, out=target)
    np.log1p(target, out=target)
    np.multiply(target, signs, out=target)


@dataclass(frozen=True)
class PreparedTreeFrame:
    tokens: FrameTokens
    model: ConfiguredTreeModelV2
    context: np.ndarray[Any, Any]
    cards: np.ndarray[Any, Any]
    resident_bytes: int
    model_bytes: int

    @classmethod
    def create(cls, tokens: FrameTokens, model: ConfiguredTreeModelV2, *, model_bytes: int = 0,
               extra_owned_bytes: int = 0) -> PreparedTreeFrame:
        import numpy as np
        if tokens.recipe_signature != signature(model.encoder.aggregate_window) or tokens.aggregate_window != model.encoder.aggregate_window:
            raise ContractError('active Tree2 token recipe/window mismatch')
        validate_frame_tokens(tokens)
        integer(extra_owned_bytes, 0, 2**63-1)
        resident = _owned((tokens, model)) + extra_owned_bytes
        resource_estimate(model, model_bytes=model_bytes, extra_bytes=resident)
        context = np.zeros(CONTEXT_WIDTH, dtype=np.float32)
        cards = np.zeros((18, CARD_WIDTH), dtype=np.float32)
        n, c = len(CARD_NUMERIC_FIELDS), len(CARD_CATEGORIES)
        for index in range(18):
            signed_log_into(cards[index, :n], tokens.card_numerics[index])
            cards[index, n:2*n] = tokens.card_present[index]
            cards[index, 2*n:2*n+c] = tokens.card_categories[index]
            cards[index, -1] = tokens.card_mask[index]
        cursor = 0
        for row, transform in ((tokens.state_numerics, True), (tokens.state_present, False),
                (tokens.state_categories, False), (tokens.state_category_present, False)):
            destination = context[cursor:cursor+len(row)]
            if transform:
                signed_log_into(destination, row)
            else:
                destination[:] = row
            cursor += len(row)
        context[cursor:cursor+cards.size] = cards.flat
        cursor += cards.size
        context[cursor:cursor+18] = tokens.opponent_learned
        cursor += 18
        signed_log_into(context[cursor:], tokens.history)
        context.flags.writeable = cards.flags.writeable = False
        return cls(tokens, model, context, cards, resident, model_bytes)

    def iter_chunks(self, candidates: CandidateTokens) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
        import numpy as np
        integer(len(candidates.kinds), 1, self.model.inference.max_candidates)
        validate_candidate_tokens(candidates, self.tokens, self.model.encoder.max_positions)
        resource_estimate(self.model, model_bytes=self.model_bytes,
                          extra_bytes=self.resident_bytes + _owned(candidates))
        width, chunk = feature_width(self.model.encoder.max_positions), self.model.inference.candidate_chunk
        for start in range(0, len(candidates.kinds), chunk):
            block = np.empty((min(chunk, len(candidates.kinds) - start), width), dtype=np.float32)
            for local, row in enumerate(block):
                index = start + local
                row[:CONTEXT_WIDTH] = self.context
                tail = row[CONTEXT_WIDTH:]
                tail[0:2] = candidates.kinds[index], candidates.targets[index]
                signed_log_into(tail[2:6], candidates.numerics[index])
                tail[6:10] = candidates.present[index]
                first, end = candidates.offsets[index:index+2]
                tail[10] = np.log1p(np.float32(end - first))
                usage = tail[11:29]
                usage.fill(0)
                for position, offset in enumerate(range(first, end)):
                    slot = candidates.slots[offset]
                    usage[slot] += np.float32(1)
                    tail[29+position*CARD_WIDTH:29+(position+1)*CARD_WIDTH] = self.cards[slot]
                signed_log_into(usage, usage)
                tail[29+(end-first)*CARD_WIDTH:] = 0
            yield start, block


def safe_tree_options_v2(value: Any, *, action_contract: Any, purpose: str, synthetic_capacity: bool) -> Json:
    from gf_models.structured_binding import check_input_binding
    raw = closed(value, {'version', 'checkpoint', 'expected_manifest_digest', 'model', 'inference', 'resources',
                         'tree_descriptor', 'feature_signature', 'action_contract', 'purpose', 'synthetic_capacity'})
    check_input_binding({key: raw[key] for key in ('action_contract', 'purpose', 'synthetic_capacity')},
                        action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    if raw['version'] != '2' or type(raw['checkpoint']) is not str or not raw['checkpoint']:
        raise ContractError('Tree2 checkpoint/version binding required')
    if type(raw['expected_manifest_digest']) is not str or re.fullmatch('[0-9a-f]{64}', raw['expected_manifest_digest']) is None:
        raise ContractError('Tree2 exact manifest digest required')
    pure = architecture_config(raw['model'])
    if raw['feature_signature'] != feature_layout(pure.encoder)['feature_signature']:
        raise ContractError('Tree2 exact feature signature differs from active ordered recipe')
    model = configured_tree_v2({**asdict(pure), 'checkpoint': raw['checkpoint'],
                               'inference': raw['inference'], 'resources': raw['resources']})
    descriptor = TreeDescriptor(**closed(raw['tree_descriptor'], set(TreeDescriptor.__dataclass_fields__)))
    resource_estimate(model, model_bytes=descriptor.model_bytes)
    return {**raw, 'model': asdict(pure), 'inference': asdict(model.inference), 'resources': asdict(model.resources),
            'tree_descriptor': asdict(descriptor)}
