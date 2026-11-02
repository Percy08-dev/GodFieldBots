"""Decisive history preservation; hashing exists only in legacy test projection."""
from dataclasses import asdict, replace

import pytest

from gf_common import structured_vocab as vocab
from gf_common.domain import ContractError, digest
from gf_common.structured import DecisionFrame, EventBank, EventDelta
from gf_common.structured_history import (
    ACTIVE_AGGREGATE,
    ACTOR_SUM_FIELDS,
    HISTORICAL_AGGREGATE,
    HISTORY_WIDTH,
    exact_history_tail,
)
from gf_common.structured_tokens import EVENT_TYPES, signature, tokenize_frame
from gf_models.config import EncoderConfig
from gf_models.encoder import _bucket, _features
from gf_models.structured_config import (
    HistoricalModelConfigV2,
    ModelConfigV2,
    architecture_identity,
    model_config_v2,
    parameter_shapes,
    recorded_architecture_identity,
    recorded_model_config_v2,
)


def tokens(value, events, *, window=128, actor=1):
    if actor == 2:
        raw = value.to_dict()
        obs = raw['observation']
        obs.update(viewer_player_id=2, current_player=2, active_player=2)
        obs['own']['player_id'], obs['opponent']['player_id'] = 2, 1
        raw['action_options']['viewer_player_id'] = 2
        value = replace(value, observation=obs, action_options=raw['action_options'])
    bank = EventBank(actor, 'own-history')
    DecisionFrame.from_input(replace(value, events=()), bank, 0)
    if events:
        bank.append(EventDelta(actor, bank.episode, 0, bank.boundary.prefix_hash, events))
    raw = value.to_dict()
    raw['observation']['state_revision'] = raw['action_options']['state_revision'] = 1
    value = replace(value, observation=raw['observation'], action_options=raw['action_options'], events=tuple(events))
    DecisionFrame.from_input(value, bank, 0)
    raw = value.to_dict()
    raw['observation']['state_revision'] = raw['action_options']['state_revision'] = 2
    value = replace(value, observation=raw['observation'], action_options=raw['action_options'])
    frame = DecisionFrame.from_input(value, bank, bank.boundary.end)
    return frame, bank, tokenize_frame(frame, bank.view(), aggregate_window=window)


@pytest.mark.parametrize('actor', [1, 2])
@pytest.mark.parametrize('window', [1, 2, 128])
def test_corrected_history_projects_every_v1_feature(make_input, actor, window):
    events = ({'type': 'artifact_used', 'player': 1, 'definition_id': 'fireball'},
              {'type': 'resource_changed', 'player': 1, 'hp_delta': 7, 'mp_delta': -3, 'money_delta': 4},
              {'type': 'artifact_used', 'player': 2, 'definition_id': 'aura'},
              {'type': 'resource_changed', 'player': 2, 'hp_delta': -7, 'mp_delta': 3, 'money_delta': -4})
    value = make_input()
    frame, _, encoded = tokens(value, events, actor=actor, window=window)
    original = replace(value, observation=frame.observation, events=events)
    old = _features(original, {'type': 'pray'}, EncoderConfig(history='public_aggregate', history_window=window))
    projected = {'history.count': encoded.history[62 + len(vocab.DEFINITIONS)],
                 'history.window_truncated': encoded.history[63 + len(vocab.DEFINITIONS)]}
    projected.update(('history.type.' + tag, encoded.history[index]) for index, tag in enumerate(EVENT_TYPES))
    for role_index, role in enumerate(('own', 'opponent')):
        offset = 64 + len(vocab.DEFINITIONS) + 8 * role_index
        for index, name in enumerate(('count', *ACTOR_SUM_FIELDS)):
            projected['history.' + role + '.' + name] = encoded.history[offset + index]
    buckets = [0] * 32
    for name, count in zip(vocab.DEFINITIONS, encoded.history[62:62 + len(vocab.DEFINITIONS)], strict=True):
        buckets[_bucket(name, 32)] += count
    projected.update((f'history.definitions.{i}', count) for i, count in enumerate(buckets))
    assert projected == {key: value for key, value in old.items() if key.startswith('history.')}
    assert HISTORY_WIDTH == len(encoded.history) == 376


def test_definition_collision_and_actor_association_survive_identical_old62(make_input):
    collision = next((a, b) for index, a in enumerate(vocab.DEFINITIONS) for b in vocab.DEFINITIONS[index + 1:]
                     if _bucket(a, 32) == _bucket(b, 32))
    first = tokens(make_input(), ({'type': 'artifact_used', 'player': 1, 'definition_id': collision[0]},))[2]
    second = tokens(make_input(), ({'type': 'artifact_used', 'player': 1, 'definition_id': collision[1]},))[2]
    assert first.interval_numerics == second.interval_numerics == ()
    assert first.history[:62] == second.history[:62] and first.history != second.history
    a = ({'type': 'resource_changed', 'player': 1, 'hp_delta': 5, 'mp_delta': 0, 'money_delta': 0},
         {'type': 'resource_changed', 'player': 2, 'hp_delta': -5, 'mp_delta': 0, 'money_delta': 0})
    b = tuple({**event, 'hp_delta': -event['hp_delta']} for event in a)
    left, right = tokens(make_input(), a)[2], tokens(make_input(), b)[2]
    assert left.history[:62] == right.history[:62] and left.history != right.history


def test_exact_key_presence_fallback_null_and_conflicting_roles():
    events = ({'player': None, 'attacker': 1, 'amount': 100},
              {'attacker': None, 'active_player': 1, 'amount': 200},
              {'attacker': 1, 'active_player': 2, 'amount': -3},
              {'player': 2, 'attacker': 1, 'amount': 4}, {'active_player': 1, 'amount': 2})
    tail = exact_history_tail(events, 1, prefix_count=5, window=128)
    offset = len(vocab.DEFINITIONS) + 2
    assert tail[offset:offset + 2] == (2, -1)
    assert tail[offset + 8:offset + 10] == (1, 4)
    assert exact_history_tail((), 2, prefix_count=0, window=128)[len(vocab.DEFINITIONS):] == (0,) * 18


def test_future_bank_and_dream_do_not_rewrite_event_time_ids(make_input):
    frame, bank, original = tokens(make_input(), ({'type': 'artifact_used', 'player': 1, 'definition_id': 'fireball'},))
    bank.append(EventDelta(1, bank.episode, bank.boundary.end, bank.boundary.prefix_hash,
        [{'type': 'artifact_used', 'player': 2, 'definition_id': 'aura'}]))
    assert tokenize_frame(frame, bank.view()).history == original.history
    # A past event's ID requires no current held definition or catalog value.
    assert original.history[62 + vocab.DEFINITIONS.index('fireball')] == 1
    assert original.history[62 + vocab.DEFINITIONS.index('aura')] == 0


def test_actual_dream_changes_current_perception_but_preserves_older_event_identity(make_input):
    event = {'type': 'artifact_used', 'player': 1, 'definition_id': 'bronze-club'}
    normal_frame, _, normal = tokens(make_input(hand=('bronze-club',)), (event,))
    dream_value = make_input(hand=('angel-axe',))
    obs = dream_value.to_dict()['observation']
    obs['own']['status']['dream'] = True
    dream_value = replace(dream_value, observation=obs)
    dream_frame, dream_bank, dream = tokens(dream_value, (event,))
    assert normal_frame.observation['own']['status']['dream'] is False
    assert dream_frame.observation['own']['status']['dream'] is True
    assert dream_frame.raw['catalog'] != normal_frame.raw['catalog']
    assert dream.card_categories[0] != normal.card_categories[0]
    assert dream.card_numerics[0] != normal.card_numerics[0]
    assert dream.interval_categories == normal.interval_categories == ()
    assert dream.history == normal.history
    assert dream.history[62 + vocab.DEFINITIONS.index('bronze-club')] == 1
    assert dream.history[62 + vocab.DEFINITIONS.index('angel-axe')] == 0
    dream_bank.append(EventDelta(1, dream_bank.episode, dream_bank.boundary.end, dream_bank.boundary.prefix_hash,
        [{'type': 'artifact_used', 'player': 2, 'definition_id': 'angel-axe'}]))
    assert tokenize_frame(dream_frame, dream_bank.view()).history == dream.history
    foreign = EventBank(2, dream_bank.episode)
    with pytest.raises(ContractError):
        tokenize_frame(dream_frame, foreign.view())


def test_exact_two_config_descriptors_old_serialization_and_early_active_reject():
    current = asdict(ModelConfigV2())
    assert current['event']['aggregate_recipe'] == ACTIVE_AGGREGATE
    old = {**current, 'event': {key: value for key, value in current['event'].items() if key != 'aggregate_recipe'}}
    old_signature = signature(128, aggregate_recipe=HISTORICAL_AGGREGATE)
    descriptor = recorded_model_config_v2(old, old_signature)
    assert type(descriptor) is HistoricalModelConfigV2 and asdict(descriptor) == old
    raw = {key: value for key, value in old.items() if key != 'resources'}
    assert recorded_architecture_identity(descriptor) == digest({'model': raw, 'token_signature': old_signature})
    assert parameter_shapes(descriptor)['frame_projection.weight'][1] + 314 == parameter_shapes(ModelConfigV2())['frame_projection.weight'][1]
    with pytest.raises(ContractError):
        model_config_v2(old)
    with pytest.raises(ContractError):
        architecture_identity(descriptor)
    with pytest.raises(ContractError):
        recorded_model_config_v2(old, signature())
    with pytest.raises(ContractError):
        recorded_model_config_v2(current, old_signature)
