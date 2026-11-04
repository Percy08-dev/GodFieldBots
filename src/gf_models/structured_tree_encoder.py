"""Deterministic FP32 Tree2 rows; current raw tokens only, no learned encoders."""
from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    import numpy as np

from gf_common.domain import ContractError, Json, closed, digest, integer
from gf_common.structured_history import APPENDED_HISTORY_FIELDS, appended_history_fields
from gf_common.structured_tokens import (
    EVENT_ACTORS,
    EVENT_NUMERIC_FIELDS,
    EVENT_TYPES,
    CandidateTokens,
    FrameTokens,
    signature,
    validate_candidate_tokens,
    validate_frame_tokens,
)
from gf_models.structured_tree_config import (
    TREE2_CONTEXT,
    TREE3_CONTEXT,
    TREE_NUMERICS,
    ConfiguredTreeModelV2,
    ConfiguredTreeModelV3,
    TreeDescriptor,
    TreeEncoderConfigV2,
    TreeEncoderConfigV3,
    TreeFeatureContext,
    architecture_config,
    configured_tree_v2,
    require_tree_context,
    resource_estimate,
    tree_context,
)

STATE_CATEGORY_FIELDS = ('phase', 'required_choice', 'first_player', 'active_player', 'current_player',
                         'result', 'own.disease', 'own.guardian', 'opponent.disease', 'opponent.guardian')
CARD_CATEGORY_FIELDS = ('definition', 'ability', 'kind', 'attribute', 'disease', 'guardian', 'source')
HISTORY_FIELDS = (tuple('event.' + name for name in EVENT_TYPES)
    + tuple('actor.' + field + '.' + role for field in EVENT_ACTORS for role in ('self', 'opponent'))
    + tuple('numeric.' + field + '.' + stat for field in EVENT_NUMERIC_FIELDS for stat in ('sum', 'present'))
    + APPENDED_HISTORY_FIELDS)


CARD_CATEGORY_FIELDS_V3 = (*CARD_CATEGORY_FIELDS[:6], 'instance_kind', 'miracle_state',
                           'definition_class', 'miracle_subclass', 'membership')


def _encoder_context(encoder: TreeEncoderConfigV2) -> TreeFeatureContext:
    if type(encoder) is TreeEncoderConfigV2:
        return TREE2_CONTEXT
    if type(encoder) is TreeEncoderConfigV3:
        return TREE3_CONTEXT
    raise ContractError('closed Tree2/Tree3 encoder required')


def categorical_limits(encoder: TreeEncoderConfigV2) -> tuple[int, ...]:
    if type(encoder) is not TreeEncoderConfigV2:
        raise ContractError('closed Tree2 encoder required')
    return _categorical_limits(encoder, TREE2_CONTEXT)


def categorical_limits_v3(encoder: TreeEncoderConfigV2) -> tuple[int, ...]:
    if type(encoder) is not TreeEncoderConfigV3:
        raise ContractError('closed Tree3 encoder required')
    return _categorical_limits(encoder, TREE3_CONTEXT)


def _categorical_limits(encoder: TreeEncoderConfigV2, context: TreeFeatureContext) -> tuple[int, ...]:
    layout = require_tree_context(context).layout
    return (tuple(len(values) + 1 for values in layout.state_categories)
        + tuple(len(values) + 1 for _ in range(18) for values in layout.card_categories)
        + (len(layout.card_categories[0]) + 1,) * 18
        + (len(layout.action_kinds) + 1, len(layout.targets) + 1)
        + tuple(len(values) + 1 for _ in range(encoder.max_positions) for values in layout.card_categories))


def escape_feature_leaf(value: str) -> str:
    """Injective LightGBM-safe leaf spelling; hierarchy separators are added later."""
    return ''.join(chr(byte) if (65 <= byte <= 90 or 97 <= byte <= 122
                                or 48 <= byte <= 57 or byte in (95, 45))
                   else f'%{byte:02X}' for byte in value.encode('utf-8'))


def _feature_fields(encoder: TreeEncoderConfigV2, context: TreeFeatureContext
                    ) -> tuple[tuple[tuple[str, ...], str], ...]:
    layout = require_tree_context(context).layout
    card_categories = CARD_CATEGORY_FIELDS if context is TREE2_CONTEXT else CARD_CATEGORY_FIELDS_V3
    history_fields = (tuple('event.' + name for name in layout.event_types)
        + tuple('actor.' + field + '.' + role for field in EVENT_ACTORS for role in ('self', 'opponent'))
        + tuple('numeric.' + field + '.' + stat for field in layout.event_numeric_fields for stat in ('sum', 'present'))
        + appended_history_fields(layout.event_categories[1]))
    if len(history_fields) != layout.history_width:
        raise ContractError('Tree history field layout mismatch')

    def card(prefix: tuple[str, ...]) -> tuple[tuple[tuple[str, ...], str], ...]:
        return (tuple(((*prefix, 'numeric', name), 'numeric') for name in layout.card_numeric_fields)
            + tuple(((*prefix, 'present', name), 'presence') for name in layout.card_numeric_fields)
            + tuple(((*prefix, 'category', name), 'category') for name in card_categories)
            + (((*prefix, 'mask'), 'presence'),))

    return (tuple((('state', 'numeric', name), 'numeric') for name in layout.state_numeric_fields)
        + tuple((('state', 'present', name), 'presence') for name in layout.state_numeric_fields)
        + tuple((('state', 'category', name), 'category') for name in STATE_CATEGORY_FIELDS)
        + tuple((('state', 'category_present', name), 'presence') for name in STATE_CATEGORY_FIELDS)
        + tuple(item for slot in range(18) for item in card(('own', str(slot))))
        + tuple((('opponent', 'learned', str(slot)), 'category') for slot in range(18))
        + tuple((('history', name), 'numeric') for name in history_fields)
        + ((('candidate', 'kind'), 'category'), (('candidate', 'target'), 'category'))
        + tuple((('candidate', 'numeric', name), 'numeric') for name in ('hp', 'mp', 'money', 'accept'))
        + tuple((('candidate', 'present', name), 'presence') for name in ('hp', 'mp', 'money', 'accept'))
        + ((('candidate', 'length'), 'numeric'),)
        + tuple((('candidate', 'used_slot', str(slot)), 'numeric') for slot in range(18))
        + tuple(item for position in range(encoder.max_positions)
                for item in card(('candidate', 'position', str(position)))))


def feature_layout(encoder: TreeEncoderConfigV2) -> Json:
    if type(encoder) is not TreeEncoderConfigV2:
        raise ContractError('closed Tree2 encoder required')
    return _feature_layout(encoder, TREE2_CONTEXT)


def feature_layout_v3(encoder: TreeEncoderConfigV2) -> Json:
    if type(encoder) is not TreeEncoderConfigV3:
        raise ContractError('closed Tree3 encoder required')
    return _feature_layout(encoder, TREE3_CONTEXT)


def _feature_layout(encoder: TreeEncoderConfigV2, context: TreeFeatureContext) -> Json:
    if _encoder_context(encoder) is not require_tree_context(context):
        raise ContractError('Tree encoder/context mismatch')
    fields = _feature_fields(encoder, context)
    names = ['.'.join(segments if context is TREE2_CONTEXT
                     else tuple(escape_feature_leaf(segment) for segment in segments))
             for segments, _ in fields]
    if len(fields) != context.feature_width(encoder.max_positions) or len(set(names)) != len(names):
        raise ContractError('Tree feature layout mismatch')
    layout = {'recipe': context.recipe, 'numeric_transform': TREE_NUMERICS, 'encoder': asdict(encoder),
              'token_signature': signature(encoder.aggregate_window, layout=context.layout),
              'feature_names': names,
              'categorical_indices': [i for i, (_, transform) in enumerate(fields) if transform == 'category'],
              'transforms': [transform for _, transform in fields]}
    if context is TREE3_CONTEXT:
        from gf_common.card_instance import corrected_card_instance_contract
        from gf_common.card_instance_rules import card_instance_rules, rule_signature
        layout.update(input_recipe=context.layout.input_recipe,
            policy_input_version=context.layout.policy_input_version,
            action_contract=asdict(context.layout.action_contract),
            layout_signature=signature(layout=context.layout),
            rule_signature=rule_signature(corrected_card_instance_contract(), card_instance_rules()),
            categorical_limits=list(_categorical_limits(encoder, context)),
            raw_field_mapping=[{'segments': list(segments), 'name': name}
                               for (segments, _), name in zip(fields, names, strict=True)])
    return {**layout, 'feature_signature': digest(layout)}


def _owned(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if type(value) is PreparedTreeFrameV3:
        size += value._owned_children(seen)
    elif isinstance(value, (tuple, list)):
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


def _prepare_frame(tokens: FrameTokens, model: ConfiguredTreeModelV2 | ConfiguredTreeModelV3, *,
                   context: TreeFeatureContext, model_bytes: int, extra_owned_bytes: int
                   ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    import numpy as np
    if tree_context(model) is not require_tree_context(context):
        raise ContractError('Tree model/context mismatch')
    if (type(tokens) is not FrameTokens
            or tokens.recipe_signature != signature(model.encoder.aggregate_window, layout=context.layout)
            or tokens.aggregate_window != model.encoder.aggregate_window):
        raise ContractError(f'active Tree{model.version} token recipe/window mismatch')
    validate_frame_tokens(tokens, layout=context.layout)
    integer(extra_owned_bytes, 0, 2**63-1)
    resident = _owned((tokens, model)) + extra_owned_bytes
    resource_estimate(model, model_bytes=model_bytes, extra_bytes=resident)
    state = np.zeros(context.context_width, dtype=np.float32)
    cards = np.zeros((18, context.card_width), dtype=np.float32)
    n, c = len(context.layout.card_numeric_fields), len(context.layout.card_categories)
    for index in range(18):
        signed_log_into(cards[index, :n], tokens.card_numerics[index])
        cards[index, n:2*n] = tokens.card_present[index]
        cards[index, 2*n:2*n+c] = tokens.card_categories[index]
        cards[index, -1] = tokens.card_mask[index]
    cursor = 0
    for row, transform in ((tokens.state_numerics, True), (tokens.state_present, False),
            (tokens.state_categories, False), (tokens.state_category_present, False)):
        destination = state[cursor:cursor+len(row)]
        if transform:
            signed_log_into(destination, row)
        else:
            destination[:] = row
        cursor += len(row)
    state[cursor:cursor+cards.size] = cards.flat
    cursor += cards.size
    state[cursor:cursor+18] = tokens.opponent_learned
    cursor += 18
    signed_log_into(state[cursor:], tokens.history)
    state.flags.writeable = cards.flags.writeable = False
    return state, cards, resident


def _iter_chunks(tokens: FrameTokens, model: ConfiguredTreeModelV2 | ConfiguredTreeModelV3,
                 state: np.ndarray[Any, Any], cards: np.ndarray[Any, Any], resident_bytes: int,
                 model_bytes: int, candidates: CandidateTokens, *, context: TreeFeatureContext
                 ) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
    import numpy as np
    if tree_context(model) is not require_tree_context(context):
        raise ContractError('Tree model/context mismatch')
    if type(candidates) is not CandidateTokens:
        raise ContractError('closed Tree candidate tokens required')
    integer(len(candidates.kinds), 1, model.inference.max_candidates)
    validate_candidate_tokens(candidates, tokens, model.encoder.max_positions, layout=context.layout)
    resource_estimate(model, model_bytes=model_bytes,
                      extra_bytes=resident_bytes + _owned(candidates))
    width, chunk = context.feature_width(model.encoder.max_positions), model.inference.candidate_chunk
    card_width, context_width = context.card_width, context.context_width
    for start in range(0, len(candidates.kinds), chunk):
        block = np.empty((min(chunk, len(candidates.kinds) - start), width), dtype=np.float32)
        for local, row in enumerate(block):
            index = start + local
            row[:context_width] = state
            tail = row[context_width:]
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
                tail[29+position*card_width:29+(position+1)*card_width] = cards[slot]
            signed_log_into(usage, usage)
            tail[29+(end-first)*card_width:] = 0
        yield start, block


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
        context, cards, resident = _prepare_frame(tokens, model, context=TREE2_CONTEXT,
            model_bytes=model_bytes, extra_owned_bytes=extra_owned_bytes)
        return cls(tokens, model, context, cards, resident, model_bytes)

    def iter_chunks(self, candidates: CandidateTokens) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
        return _iter_chunks(self.tokens, self.model, self.context, self.cards, self.resident_bytes,
                            self.model_bytes, candidates, context=TREE2_CONTEXT)


class PreparedTreeFrameV3:
    """Factory-owned immutable caches bound to their exact participant inputs."""

    __slots__ = ('__binding', '__cards', '__context', '__model', '__model_bytes',
                 '__resident_bytes', '__tokens')

    __tokens: FrameTokens
    __model: ConfiguredTreeModelV3
    __context: np.ndarray[Any, Any]
    __cards: np.ndarray[Any, Any]
    __resident_bytes: int
    __model_bytes: int
    __binding: tuple[Any, ...]

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise ContractError('Tree3 prepared caches require the validated create factory')

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError('Tree3 prepared values are concrete')

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('Tree3 prepared values are immutable')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('Tree3 prepared values are immutable')

    @property
    def tokens(self) -> FrameTokens:
        return self.__tokens

    @property
    def model(self) -> ConfiguredTreeModelV3:
        return self.__model

    @property
    def context(self) -> np.ndarray[Any, Any]:
        return self.__context

    @property
    def cards(self) -> np.ndarray[Any, Any]:
        return self.__cards

    @property
    def resident_bytes(self) -> int:
        return self.__resident_bytes

    @property
    def model_bytes(self) -> int:
        return self.__model_bytes

    @classmethod
    def create(cls, tokens: FrameTokens, model: ConfiguredTreeModelV3, *, model_bytes: int = 0,
               extra_owned_bytes: int = 0) -> PreparedTreeFrameV3:
        import numpy as np
        if cls is not PreparedTreeFrameV3:
            raise ContractError('closed Tree3 prepared factory required')
        integer(extra_owned_bytes, 0, 2**63-1)
        prepared = object.__new__(cls)
        # The shared context estimate covers the temporary writable packing arrays.
        # Admit the retained immutable copy, slots/binding and an additional 64KiB
        # array-metadata allowance before allocating any numerical buffer.
        cache_bytes = (TREE3_CONTEXT.context_width + 18*TREE3_CONTEXT.card_width)*4
        overhead = (sys.getsizeof(prepared) + sys.getsizeof((None,)*8)
                    + cache_bytes + 2*sys.getsizeof(b'') + 65536)
        context, cards, resident = _prepare_frame(tokens, model, context=TREE3_CONTEXT,
            model_bytes=model_bytes, extra_owned_bytes=extra_owned_bytes + overhead)
        # Bytes backing prevents public accessors from enabling WRITEABLE again.
        immutable_context = np.ndarray(context.shape, dtype=np.float32, buffer=context.tobytes())
        immutable_cards = np.ndarray(cards.shape, dtype=np.float32, buffer=cards.tobytes())
        for name, value in (('tokens', tokens), ('model', model), ('context', immutable_context),
                            ('cards', immutable_cards), ('resident_bytes', resident),
                            ('model_bytes', model_bytes)):
            object.__setattr__(prepared, '_PreparedTreeFrameV3__' + name, value)
        object.__setattr__(prepared, '_PreparedTreeFrameV3__binding',
                           (tokens, model, immutable_context, immutable_cards, resident, model_bytes,
                            immutable_context.base, immutable_cards.base))
        return prepared

    def _owned_children(self, seen: set[int]) -> int:
        # ndarray views retain their immutable bytes owners; count these explicitly.
        return (_owned(self.__binding, seen) + _owned(self.__context.base, seen)
                + _owned(self.__cards.base, seen))

    def __check_binding(self) -> None:
        import numpy as np
        binding = self.__binding
        if (type(self) is not PreparedTreeFrameV3
                or type(binding) is not tuple or len(binding) != 8
                or self.__tokens is not binding[0] or self.__model is not binding[1]
                or self.__context is not binding[2] or self.__cards is not binding[3]
                or self.__resident_bytes != binding[4] or self.__model_bytes != binding[5]
                or type(self.__tokens) is not FrameTokens or type(self.__model) is not ConfiguredTreeModelV3
                or self.__tokens.aggregate_window != self.__model.encoder.aggregate_window
                or self.__tokens.recipe_signature != signature(
                    self.__model.encoder.aggregate_window, layout=TREE3_CONTEXT.layout)
                or self.__context.shape != (TREE3_CONTEXT.context_width,)
                or self.__cards.shape != (18, TREE3_CONTEXT.card_width)
                or self.__context.dtype != np.float32 or self.__cards.dtype != np.float32
                or not self.__context.flags.c_contiguous or not self.__cards.flags.c_contiguous
                or self.__context.flags.writeable or self.__cards.flags.writeable
                or type(self.__context.base) is not bytes or self.__context.base is not binding[6]
                or type(self.__cards.base) is not bytes or self.__cards.base is not binding[7]):
            raise ContractError('Tree3 prepared token/model/cache binding mismatch')

    def iter_chunks(self, candidates: CandidateTokens) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
        chunks = _iter_chunks(self.__tokens, self.__model, self.__context, self.__cards,
                              self.__resident_bytes, self.__model_bytes, candidates, context=TREE3_CONTEXT)
        while True:
            self.__check_binding()
            try:
                chunk = next(chunks)
            except StopIteration:
                return
            yield chunk


def safe_tree_options_v2(value: Any, *, action_contract: Any, purpose: str, synthetic_capacity: bool) -> Json:
    from gf_models.structured_binding import check_input_binding
    training_fields={'training_binding'} if type(value) is dict and 'training_binding' in value else set()
    raw = closed(value, training_fields|{'version', 'checkpoint', 'expected_manifest_digest', 'model', 'inference', 'resources',
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
    if training_fields:
        from gf_models.structured_tree_pairing import TrainingBinding
        TrainingBinding(**closed(raw['training_binding'],{'directory','request_sha256','completion_sha256'}))
    descriptor = TreeDescriptor(**closed(raw['tree_descriptor'], set(TreeDescriptor.__dataclass_fields__)))
    resource_estimate(model, model_bytes=descriptor.model_bytes)
    return {**raw, 'model': asdict(pure), 'inference': asdict(model.inference), 'resources': asdict(model.resources),
            'tree_descriptor': asdict(descriptor)}
