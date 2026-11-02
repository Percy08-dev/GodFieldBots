"""Pure ordered public history statistics, independent of models/catalog values."""
from __future__ import annotations

from gf_common import structured_vocab as vocab
from gf_common.domain import ContractError, Json, integer

ACTIVE_AGGREGATE = 'public-exact-v2'
HISTORICAL_AGGREGATE = 'historical-structured-62'
LEGACY_HISTORY_WIDTH = 62
ACTOR_SUM_FIELDS = ('amount', 'hp_delta', 'mp_delta', 'money_delta', 'mp_spent', 'price', 'attack')
APPENDED_HISTORY_FIELDS = (tuple('definition.' + name for name in vocab.DEFINITIONS)
    + ('window.count', 'window.truncated')
    + tuple(role + '.' + name for role in ('own', 'opponent') for name in ('count', *ACTOR_SUM_FIELDS)))
HISTORY_WIDTH = LEGACY_HISTORY_WIDTH + len(APPENDED_HISTORY_FIELDS)


def history_width(recipe: str) -> int:
    if recipe == ACTIVE_AGGREGATE:
        return HISTORY_WIDTH
    if recipe == HISTORICAL_AGGREGATE:
        return LEGACY_HISTORY_WIDTH
    raise ContractError('unsupported closed structured aggregate recipe')


def exact_history_tail(events: tuple[Json, ...], actor: int, *, prefix_count: int, window: int) -> tuple[int, ...]:
    """Use event-time IDs and exact key-presence fallback, never a catalog lookup.

    events is the already bounded last window of the participant's own prefix.
    Existing null player/attacker keys deliberately prevent fallback.
    """
    integer(actor, 1, 2)
    integer(window, 1, 65536)
    integer(prefix_count, len(events), 2**31 - 1)
    if len(events) != min(prefix_count, window):
        raise ContractError('exact history tail must cover the last own-prefix window')
    counts = [0] * len(vocab.DEFINITIONS)
    roles = [[0] * (1 + len(ACTOR_SUM_FIELDS)) for _ in range(2)]
    for event in events:
        if 'definition_id' in event:
            try:
                counts[vocab.DEFINITIONS.index(event['definition_id'])] += 1
            except ValueError as error:
                raise ContractError('unknown event-time public definition ID') from error
        player = event.get('player', event.get('attacker', event.get('active_player')))
        if player not in (actor, 3 - actor):
            continue
        row = roles[0 if player == actor else 1]
        row[0] += 1
        for index, field in enumerate(ACTOR_SUM_FIELDS, 1):
            row[index] += event.get(field, 0)
    return (*counts, len(events), int(prefix_count > window), *roles[0], *roles[1])
