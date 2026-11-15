"""Closed HPO correspondence facts. Values never grant Source authority."""
from __future__ import annotations

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer

SELECTION_VERSION = 'hpo-endpoint-selection-v1'
NATIVE_CAPABILITY = 'model4;collection6;proof3;numeric5;selection1;provider1'
CONTEXT_KEYS = {'selection_version', 'decision_budget', 'used_before', 'remaining', 'learning_seed', 'batch'}
SELECTION_KEYS = {'context', 'derived_seed', 'episode_counts', 'collected_decisions', 'used_decisions', 'indices', 'indices_sha256'}
PROOF_KINDS = ('card-instance-hpo-original-source-proof', 'card-instance-hpo-packed-original-source-proof')


def selection_context(*, decision_budget: int, used_before: int, learning_seed: int, batch: int) -> Json:
    integer(decision_budget, 1, 2**63 - 1)
    integer(used_before, 0, decision_budget)
    integer(learning_seed, 0, 2**63 - 1)
    integer(batch, 0, 2**63 - 1)
    return {'selection_version': SELECTION_VERSION, 'decision_budget': decision_budget,
        'used_before': used_before, 'remaining': decision_budget - used_before,
        'learning_seed': learning_seed, 'batch': batch}


def checked_context(value: Json) -> Json:
    raw = closed(value, CONTEXT_KEYS)
    expected = selection_context(**{key: raw[key] for key in ('decision_budget', 'used_before', 'learning_seed', 'batch')})
    if type(raw['remaining']) is not int or canonical(raw) != canonical(expected):
        raise ContractError('HPO selection context mismatch')
    return raw


def selection_seed(context: Json) -> int:
    return int(digest({'recipe': 'hpo-ppo-selection-seed-v1', 'context': checked_context(context)})[:32], 16)


def checked_selection(value: Json) -> Json:
    raw = closed(value, SELECTION_KEYS)
    context = checked_context(raw['context'])
    if type(raw['episode_counts']) is not list or type(raw['indices']) is not list:
        raise ContractError('HPO complete counts and selected indices must be lists')
    for count in raw['episode_counts']:
        integer(count, 0, 1024)
    count = sum(raw['episode_counts'])
    used = min(context['remaining'], count)
    integer(raw['collected_decisions'], count, count)
    integer(raw['used_decisions'], used, used)
    integer(raw['derived_seed'], 0, 2**128 - 1)
    if raw['derived_seed'] != selection_seed(context):
        raise ContractError('HPO independently derived seed differs')
    for index in raw['indices']:
        integer(index, 0, count - 1)
    if (len(raw['indices']) != used or sorted(set(raw['indices'])) != raw['indices']
            or raw['indices_sha256'] != digest(raw['indices'])):
        raise ContractError('HPO sorted selection count/hash differs')
    return raw


def selected_locations(selection: Json) -> tuple[tuple[int, int], ...]:
    raw = checked_selection(selection)
    indices = set(raw['indices'])
    locations: list[tuple[int, int]] = []
    offset = 0
    for episode, count in enumerate(raw['episode_counts']):
        locations.extend((episode, position) for position in range(count) if offset + position in indices)
        offset += count
    return tuple(locations)


def optimizer_delta(used: int, *, epochs: int, minibatch_size: int) -> int:
    integer(used, 0, 2**63 - 1)
    integer(epochs, 1, 1000)
    integer(minibatch_size, 1, 2**20)
    return epochs * ((used + minibatch_size - 1) // minibatch_size)


def require_owned_record_version(value: Json, kind: str) -> None:
    """Select producer record versions explicitly; values never grant execution."""
    versions={'collection':8,'proof':5,'numeric':6,'update':6,'generation':5,'registry':6,'cache':2}
    if kind not in versions:
        raise ContractError('unknown owned record kind')
    if (type(value) is not dict or type(value.get('schema_version')) is not int
            or value['schema_version']!=versions[kind] or value.get('origin')!='producer-owned-v1'
            or value.get('encoder')!='canonical-python'):
        raise ContractError('mixed legacy/producer-owned record versions or origin/encoder')
