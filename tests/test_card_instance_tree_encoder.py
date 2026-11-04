"""Corrected participant-only Tree3 rows against an independent scalar recipe."""
import copy
import re
import subprocess
import sys
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from gf_common import structured_vocab as vocab
from gf_common.domain import ContractError, digest
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CARD_INSTANCE_LAYOUT, CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_tokens import signature, tokenize_frame, tokenize_table
from gf_models.structured_tree_config import (
    TREE3_CONTEXT,
    ConfiguredTreeModelV2,
    ConfiguredTreeModelV3,
    TreeArchitectureV2,
    TreeEncoderConfigV2,
    TreeEncoderConfigV3,
    TreeInferenceConfigV2,
    TreeResourcesV2,
    architecture_config,
    architecture_config_v3,
    architecture_identity,
    architecture_identity_v3,
    configured_tree_v2,
    configured_tree_v3,
    feature_width_v3,
    resource_estimate,
)
from gf_models.structured_tree_encoder import (
    PreparedTreeFrame,
    PreparedTreeFrameV3,
    _owned,
    categorical_limits,
    categorical_limits_v3,
    escape_feature_leaf,
    feature_layout,
    feature_layout_v3,
)
from tests.card_instance_action_support import inputs
from tests.test_corrected_card_instance import records
from tests.test_structured_tree_encoder import independent_row
from tests.test_structured_tree_encoder import tokens as old_tokens

LAYOUT = CORRECTED_CARD_INSTANCE_LAYOUT


def table_for(frame, actions, *, synthetic=False):
    return CandidateTable.from_actions(frame, actions, action_contract=LAYOUT.action_contract,
                                       purpose='component', synthetic_capacity=synthetic)


def prepared(value, actions, *, positions=18, window=128, chunk=2):
    frame, bank, _ = records(value)
    table = table_for(frame, actions(frame) if callable(actions) else actions)
    raw = tokenize_frame(frame, bank.view(), aggregate_window=window, layout=LAYOUT)
    candidates = tokenize_table(table, frame, positions, aggregate_window=window, layout=LAYOUT)
    model = ConfiguredTreeModelV3(
        encoder=TreeEncoderConfigV3(max_positions=positions, aggregate_window=window),
        inference=TreeInferenceConfigV2(candidate_chunk=chunk))
    return frame, bank, PreparedTreeFrameV3.create(raw, model), candidates


def rows(preparation, candidates):
    return np.concatenate([block for _, block in preparation.iter_chunks(candidates)])


def scalar_log(number):
    value = np.float32(number)
    return np.sign(value) * np.log1p(np.abs(value))


def reference(raw, candidates, index, positions):
    # Independent raw-token flattening: no production widths, layouts or packing helpers.
    def numeric(values):
        return [scalar_log(number) for number in values]

    def card(slot):
        return [*numeric(raw.card_numerics[slot]), *raw.card_present[slot],
                *raw.card_categories[slot], raw.card_mask[slot]]

    chosen = candidates.slots[candidates.offsets[index]:candidates.offsets[index+1]]
    return np.asarray([
        *numeric(raw.state_numerics), *raw.state_present, *raw.state_categories,
        *raw.state_category_present, *[value for slot in range(18) for value in card(slot)],
        *raw.opponent_learned, *numeric(raw.history), candidates.kinds[index], candidates.targets[index],
        *numeric(candidates.numerics[index]), *candidates.present[index], scalar_log(len(chosen)),
        *numeric([chosen.count(slot) for slot in range(18)]),
        *[value for slot in chosen for value in card(slot)], *([0] * (64*(positions-len(chosen))))
    ], dtype=np.float32)


@pytest.mark.parametrize('kind', LAYOUT.action_kinds)
@pytest.mark.parametrize('positions,window', [(18, 1), (128, 128), (18, 1024)])
def test_all_action_families_independent_fp32_rows_and_selected_full_equality(kind, positions, window):
    choice = 'defense' if kind == 'respond' else 'buy_accept_reject' if kind == 'buy_decision' else 'turn_action'
    value = inputs([('bronze-club', None), ('fireball', 'unused'), ('fireball', 'learned')], choice=choice)

    def actions(frame):
        a, b, c = frame.references
        extra = {
            'play_sequence': {'ordered_card_refs': [c, a, b], 'target': 2},
            'respond': {'ordered_card_refs': [b, a]},
            'discard': {'card_refs': [c, a]}, 'sacrifice': {'card_refs': [b, c]},
            'exchange': {'card_ref': a, 'hp': -1, 'mp': 0, 'money': 1},
            'sell': {'card_ref': a, 'sold_card_ref': c, 'target': 1},
            'buy': {'card_ref': b, 'target': 2}, 'buy_decision': {'accept': False},
            'invoke_phenomenon': {'card_ref': a}, 'pray': {},
        }
        return [{'type': kind, **extra[kind]}] * 3

    _, _, preparation, candidates = prepared(value, actions, positions=positions, window=window)
    blocks = list(preparation.iter_chunks(candidates))
    assert [offset for offset, _ in blocks] == [0, 2]
    actual = np.concatenate([block for _, block in blocks])
    expected = np.stack([reference(preparation.tokens, candidates, i, positions) for i in range(3)])
    assert actual.shape == (3, 1658 + 64*positions)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous and np.isfinite(actual).all()
    assert actual.tobytes() == expected.tobytes()
    # A one-row selected projection preserves exactly the original ordered row.
    first, end = candidates.offsets[1:3]
    selected = replace(candidates, kinds=(candidates.kinds[1],), targets=(candidates.targets[1],),
        numerics=(candidates.numerics[1],), present=(candidates.present[1],),
        offsets=(0, end-first), slots=candidates.slots[first:end])
    assert rows(preparation, selected)[0].tobytes() == actual[1].tobytes()


@pytest.mark.parametrize('count', [0, 7, 12, 18])
def test_empty_and_duplicate_learned_possessions_opponent_order_and_masks(count):
    value = inputs([('fireball', 'learned')] * count)
    value.observation['opponent']['learned_miracles'] = ['aura', 'aura', 'fireball']
    _, _, preparation, candidates = prepared(value, [{'type': 'pray'}])
    actual = rows(preparation, candidates)[0]
    assert actual.tobytes() == reference(preparation.tokens, candidates, 0, 18).tobytes()
    names = feature_layout_v3(preparation.model.encoder)['feature_names']
    assert preparation.tokens.card_mask == (True,) * count + (False,) * (18-count)
    assert preparation.tokens.opponent_learned[0] == preparation.tokens.opponent_learned[1]
    assert preparation.tokens.opponent_learned[1] != preparation.tokens.opponent_learned[2]
    assert actual[names.index('opponent.learned.0')] == preparation.tokens.opponent_learned[0]
    for slot in range(count):
        assert actual[names.index(f'own.{slot}.numeric.possession_position')] == scalar_log(slot+1)
        assert actual[names.index(f'own.{slot}.category.miracle_state')] == 3
    assert np.all(actual[-18*64:] == 0)  # empty candidate positions, independently of own possession


def test_ref_rename_candidate_and_slot_order_are_distinct_from_numeric_identity():
    value = inputs([('bronze-club', None), ('fireball', 'unused'), ('fireball', 'learned')])
    def actions(frame):
        return [
        {'type': 'play_sequence', 'ordered_card_refs': list(frame.references), 'target': 1},
        {'type': 'play_sequence', 'ordered_card_refs': list(frame.references[::-1]), 'target': 2},
        {'type': 'pray'}]
    _, _, preparation, candidates = prepared(value, actions)
    original = rows(preparation, candidates)
    renamed = copy.deepcopy(value)
    mapping = {card['card_ref']: 'c1_' + format(2**128-i-1, '032x')
               for i, card in enumerate(renamed.observation['own']['cards'])}
    for card in renamed.observation['own']['cards']:
        card['card_ref'] = mapping[card['card_ref']]
    for card in renamed.action_options['action_options']['usable_cards']:
        card['card_ref'] = mapping[card['card_ref']]
    renamed.action_options['action_options']['candidate_groups'] = {'opaque-group-name': copy.deepcopy(renamed.action_options['action_options']['usable_cards'])}
    _, _, other, other_candidates = prepared(renamed, actions)
    assert other.tokens.frame_hash != preparation.tokens.frame_hash
    assert rows(other, other_candidates).tobytes() == original.tobytes()
    _, _, reverse, reverse_candidates = prepared(value, lambda frame: actions(frame)[::-1], chunk=1)
    assert rows(reverse, reverse_candidates).tobytes() == original[::-1].tobytes()
    names = feature_layout_v3(preparation.model.encoder)['feature_names']
    assert original[0, names.index('candidate.position.0.category.instance_kind')] == 2
    assert original[1, names.index('candidate.position.0.category.miracle_state')] == 3
    # All slots used once, but ordered card positions are different.
    assert original[0, names.index('candidate.used_slot.0')] == original[1, names.index('candidate.used_slot.0')]


def test_fog_missingness_guardian_namespaces_and_frozen_own_prefix():
    value = inputs([('angel-axe', None)])
    value.observation['own']['status'].update(fog=True, dream=True)
    value.observation['opponent'] = {'player_id': 2, 'learned_miracles': ['aura', 'aura'], 'ascended': False}
    value = replace(value, events=(
        {'type': 'guardian_set', 'player': 2, 'definition_id': vocab.GUARDIANS[0]},
        {'type': 'artifact_used', 'player': 1, 'definition_id': 'bronze-club'},
        {'type': 'resource_changed', 'player': 1, 'hp_delta': -3, 'mp_delta': 0, 'money_delta': 0},
    ))
    frame, bank, preparation, candidates = prepared(value, [{'type': 'pray'}])
    actual = rows(preparation, candidates)[0]
    names = feature_layout_v3(preparation.model.encoder)['feature_names']
    assert actual[names.index('state.present.opponent%2Ehp')] == 0
    assert actual[names.index('state.category.opponent%2Eguardian')] == 1
    assert actual[names.index('state.category_present.opponent%2Eguardian')] == 0
    guardian_name = 'history.definition%2E' + escape_feature_leaf(vocab.GUARDIANS[0])
    assert actual[names.index(guardian_name)] == scalar_log(1)
    assert actual[names.index('history.definition%2Ebronze-club')] == scalar_log(1)
    assert actual[names.index('own.0.category.definition')] == vocab.DEFINITIONS.index('angel-axe')+2
    before = preparation.tokens
    bank.append(event_delta(1, bank.episode, bank.boundary.end, bank.boundary.prefix_hash,
        [{'type': 'safe', 'player': 2}], layout=LAYOUT))
    assert tokenize_frame(frame, bank.view(), layout=LAYOUT) == before
    assert np.isfinite(actual).all() and actual.tobytes() == reference(before, candidates, 0, 18).tobytes()


def test_layout_metadata_escaped_leaf_mapping_categories_and_complete_signature():
    encoder = TreeEncoderConfigV3()
    layout = feature_layout_v3(encoder)
    assert (TREE3_CONTEXT.card_width, TREE3_CONTEXT.context_width, feature_width_v3(18)) == (64, 1629, 2810)
    assert layout['token_signature'] == layout['layout_signature'] == 'ed0c61152b78fa984aad342be22c6b9d6ef49b12b8f0d1c0f0de89724468002f'
    assert layout['rule_signature'] == '3f55f044225fb20c63d683145cdceb397c36f0b7161c7b16793f72210db3e86d'
    assert layout['policy_input_version'] == 'player-card-refs-v2'
    assert layout['input_recipe'] == 'participant-structured-v2'
    assert layout['action_contract'] == asdict(LAYOUT.action_contract)
    assert len(layout['feature_names']) == 2810
    assert len(layout['categorical_indices']) == 10 + 18*11 + 18 + 2 + 18*11
    assert tuple(layout['categorical_limits']) == categorical_limits_v3(encoder)
    assert all(re.fullmatch('[A-Za-z0-9_.%\\-]+', name) for name in layout['feature_names'])
    assert 'own.0.numeric.role%2Emiracle%3Aweapon' in layout['feature_names']
    assert {'segments': ['own', '0', 'numeric', 'role.miracle:weapon'],
            'name': 'own.0.numeric.role%2Emiracle%3Aweapon'} in layout['raw_field_mapping']
    assert escape_feature_leaf('é.%:[]') == '%C3%A9%2E%25%3A%5B%5D'
    assert len({escape_feature_leaf(s) for s in ('a.b', 'a%2Eb', 'a:b', 'a%3Ab')}) == 4
    assert digest({key: value for key, value in layout.items() if key != 'feature_signature'}) == layout['feature_signature']
    assert feature_layout_v3(replace(encoder, aggregate_window=1))['feature_signature'] != layout['feature_signature']


def test_closed_model3_roundtrip_and_legacy_defaults_fields_identities_stay_exact():
    model = ConfiguredTreeModelV3()
    assert configured_tree_v3(asdict(model)) == model
    assert architecture_config_v3(asdict(model.architecture())) == model.architecture()
    assert [f.name for f in fields(model)] == [f.name for f in fields(ConfiguredTreeModelV2())]
    other = replace(model, checkpoint='elsewhere', inference=TreeInferenceConfigV2(candidate_chunk=1))
    assert architecture_identity_v3(model.architecture()) == architecture_identity_v3(other.architecture())
    assert digest(asdict(ConfiguredTreeModelV2())) == '4ab670d4e29c10041b79517a6ce876735f6de35ddb272d076d997a101bd88f8b'
    assert architecture_identity(TreeArchitectureV2()) == '92bf2f93e061a8a9b15d67690b613c18aa3a39e5efe850756ae7077a0a2aaa06'
    assert resource_estimate(ConfiguredTreeModelV2()) == {
        'feature_width': 1647, 'array_bytes': 1686528, 'fixed_bytes': 335544320,
        'context_bytes': 72008, 'working_bytes': 5067776, 'total_bytes': 340684104}
    for parser, raw in ((configured_tree_v2, asdict(model)), (configured_tree_v3, asdict(ConfiguredTreeModelV2())),
                        (architecture_config, asdict(model.architecture())),
                        (architecture_config_v3, asdict(TreeArchitectureV2()))):
        with pytest.raises(ContractError):
            parser(raw)
    for bad in ({'version': '2'}, {'unexpected': {}}, {'encoder': asdict(TreeEncoderConfigV2())}):
        with pytest.raises(ContractError):
            configured_tree_v3({**asdict(model), **bad})
    for operation in (lambda: feature_layout(model.encoder), lambda: categorical_limits(model.encoder),
                      lambda: feature_layout_v3(TreeEncoderConfigV2()),
                      lambda: ConfiguredTreeModelV3(encoder=TreeEncoderConfigV2())):
        with pytest.raises(ContractError):
            operation()
    code = ('import sys; from gf_models.structured_tree_config import ConfiguredTreeModelV3, architecture_identity_v3; '
            'architecture_identity_v3(ConfiguredTreeModelV3().architecture()); '
            "assert not ({'numpy','torch','lightgbm','pyarrow'} & sys.modules.keys())")
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('positions,window,expected', [
    (18, 128, '4842bcaafe3e2e15646c9c4c4f496a86525e5df3bf4a9c228b48bfa71c53dd5a'),
    (128, 1, '69f0b00add8db20ffad24d1f2c5c0f4d2900903350480ed24b6b9e5799355c3d'),
    (18, 1024, '3e571bc2139d0afbfb0d7b1922851d5890569106383a751623b1b824b950d485')])
def test_frozen_tree2_feature_signatures_and_full_fp32_bytes(make_input, positions, window, expected):
    raw, candidates = old_tokens(make_input, positions=positions, window=window)
    model = ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(max_positions=positions, aggregate_window=window))
    actual = rows(PreparedTreeFrame.create(raw, model), candidates)
    assert actual.tobytes() == np.stack([independent_row(raw, candidates, i, positions) for i in range(4)]).tobytes()
    assert feature_layout(model.encoder)['feature_signature'] == expected


def test_resource_admission_and_mixed_or_malformed_tokens_refuse_before_array_allocation(monkeypatch):
    _, _, preparation, candidates = prepared(inputs([('fireball', 'learned')]), [{'type': 'pray'}])
    old_frame, old_bank, _ = records(inputs([]), layout=CARD_INSTANCE_LAYOUT)
    old_raw = tokenize_frame(old_frame, old_bank.view(), layout=CARD_INSTANCE_LAYOUT)
    with pytest.raises(ContractError):
        replace(preparation.model, inference=TreeInferenceConfigV2(candidate_chunk=65536))
    assert resource_estimate(preparation.model)['array_bytes'] == 2*2810*4
    monkeypatch.setattr(np, 'zeros', lambda *a, **kw: pytest.fail('refuse before context allocation'))
    for raw, model, loader in (
        (preparation.tokens, preparation.model, PreparedTreeFrame),
        (preparation.tokens, ConfiguredTreeModelV2(), PreparedTreeFrameV3),
        (old_raw, preparation.model, PreparedTreeFrameV3),
        (replace(preparation.tokens, card_mask=(False,)*18), preparation.model, PreparedTreeFrameV3),
    ):
        with pytest.raises(ContractError):
            loader.create(raw, model)
    total = resource_estimate(preparation.model, model_bytes=1024**2,
        extra_bytes=_owned((preparation.tokens, preparation.model)))['total_bytes']
    bounded = replace(preparation.model, resources=TreeResourcesV2(max_owned_bytes=total-1024))
    with pytest.raises(ContractError, match='resource admission'):
        PreparedTreeFrameV3.create(preparation.tokens, bounded, model_bytes=1024**2)
    monkeypatch.setattr(np, 'empty', lambda *a, **kw: pytest.fail('refuse before candidate allocation'))
    for bad in (replace(candidates, kinds=(), targets=(), numerics=(), present=(), offsets=(0,), slots=()),
                replace(candidates, recipe_signature=signature()),
                replace(candidates, offsets=(0, 1), slots=(17,)),
                replace(candidates, kinds=(True,))):
        with pytest.raises(ContractError):
            next(preparation.iter_chunks(bad))


def test_explicit_synthetic128_order_counts_and_selected_distinct_row_parity():
    frame, bank, _ = records(inputs([('fireball', 'learned'), ('bronze-club', None)]))
    refs = [frame.references[i % 2] for i in range(128)]
    table = table_for(frame, [
        {'type': 'pray'},
        {'type': 'play_sequence', 'ordered_card_refs': refs, 'target': 2},
        {'type': 'play_sequence', 'ordered_card_refs': refs[::-1], 'target': 1},
    ], synthetic=True)
    with pytest.raises(ContractError):
        table.to_action(1, frame)
    model = ConfiguredTreeModelV3(encoder=TreeEncoderConfigV3(max_positions=128),
                                inference=TreeInferenceConfigV2(candidate_chunk=2))
    raw = tokenize_frame(frame, bank.view(), layout=LAYOUT)
    candidates = tokenize_table(table, frame, 128, layout=LAYOUT)
    preparation = PreparedTreeFrameV3.create(raw, model)
    actual = rows(preparation, candidates)
    assert actual.tobytes() == np.stack([reference(raw, candidates, i, 128) for i in range(3)]).tobytes()
    selected = replace(candidates, kinds=(candidates.kinds[2],), targets=(candidates.targets[2],),
        numerics=(candidates.numerics[2],), present=(candidates.present[2],),
        offsets=(0, 128), slots=candidates.slots[128:])
    assert rows(preparation, selected)[0].tobytes() == actual[2].tobytes()
    names = feature_layout_v3(model.encoder)['feature_names']
    assert actual[1, names.index('candidate.used_slot.0')] == scalar_log(64)
    assert actual[1, names.index('candidate.used_slot.1')] == scalar_log(64)
    assert actual[1, names.index('candidate.position.0.category.instance_kind')] == 3
    assert actual[2, names.index('candidate.position.0.category.instance_kind')] == 2


def test_real_model_and_candidate_ownership_are_carried_into_resource_admission(monkeypatch):
    _, _, preparation, candidates = prepared(inputs([('bronze-club', None)]), [{'type': 'pray'}])
    model = preparation.model
    native_bytes = 1024**2
    total = resource_estimate(model, model_bytes=native_bytes,
                             extra_bytes=preparation.resident_bytes)['total_bytes']
    bounded = replace(model, resources=TreeResourcesV2(max_owned_bytes=total + _owned(candidates)//2))
    admitted = PreparedTreeFrameV3.create(preparation.tokens, bounded, model_bytes=native_bytes)
    assert admitted.model_bytes == native_bytes
    monkeypatch.setattr(np, 'empty', lambda *a, **kw: pytest.fail('candidate owner must reject before allocation'))
    with pytest.raises(ContractError, match='resource admission'):
        next(admitted.iter_chunks(candidates))
    with pytest.raises(ContractError, match='resource admission'):
        replace(model, resources=TreeResourcesV2(max_array_bytes=2*2810*4-1))


def actor2_preparation():
    value = inputs([('fireball', 'learned')])
    value.observation.update(viewer_player_id=2, current_player=2, active_player=2)
    value.observation['own']['player_id'] = 2
    value.observation['opponent']['player_id'] = 1
    value.action_options['viewer_player_id'] = 2
    value.action_options['action_options']['actor'] = 2
    bank = EventBank(2, 'tree3-second-viewer', layout=LAYOUT)
    frame = DecisionFrame.from_input(value, bank, 0, layout=LAYOUT)
    raw = tokenize_frame(frame, bank.view(), layout=LAYOUT)
    candidates = tokenize_table(table_for(frame, [{'type': 'pray'}]), frame, 18, layout=LAYOUT)
    return PreparedTreeFrameV3.create(raw, ConfiguredTreeModelV3()), candidates


def test_factory_bound_actor_and_window_cannot_be_replaced_or_directly_constructed(monkeypatch):
    value = replace(inputs([('bronze-club', None)]), events=({'type': 'safe', 'player': 1},)*3)
    _, _, first, candidates = prepared(value, [{'type': 'pray'}])
    second, second_candidates = actor2_preparation()
    _, _, short, short_candidates = prepared(value, [{'type': 'pray'}], window=1)
    names = feature_layout_v3(first.model.encoder)['feature_names']
    history = names.index('history.event%2Esafe')
    assert rows(first, candidates)[0, history] == scalar_log(3)
    assert rows(short, short_candidates)[0, history] == scalar_log(1)
    assert rows(second, second_candidates)[0].tobytes() == reference(second.tokens, second_candidates, 0, 18).tobytes()
    assert first.cards[0].tobytes() != second.cards[0].tobytes()
    monkeypatch.setattr(np, 'empty', lambda *a, **kw: pytest.fail('no row allocation for replacement'))
    monkeypatch.setattr(np, 'zeros', lambda *a, **kw: pytest.fail('no cache allocation for replacement'))
    for changes in ({'tokens': second.tokens}, {'model': short.model},
                    {'tokens': second.tokens, 'model': second.model}):
        with pytest.raises(TypeError):
            replace(first, **changes)
        with pytest.raises(ContractError, match='create factory'):
            PreparedTreeFrameV3(**{
                'tokens': first.tokens, 'model': first.model, 'context': first.context,
                'cards': first.cards, 'resident_bytes': first.resident_bytes,
                'model_bytes': first.model_bytes, **changes})
        for name, replacement in changes.items():
            with pytest.raises(AttributeError):
                setattr(first, name, replacement)
    # Candidate identity remains checked even when both actors use the same window.
    with pytest.raises(ContractError, match='binding mismatch'):
        next(first.iter_chunks(second_candidates))
    with pytest.raises(ContractError, match='binding mismatch'):
        next(first.iter_chunks(short_candidates))


@pytest.mark.parametrize('replacement', ['actor', 'window', 'cache'])
def test_iteration_rechecks_private_exact_binding_before_row_allocation(monkeypatch, replacement):
    _, _, first, candidates = prepared(inputs([('bronze-club', None)]), [{'type': 'pray'}])
    second, second_candidates = actor2_preparation()
    if replacement == 'actor':
        object.__setattr__(first, '_PreparedTreeFrameV3__tokens', second.tokens)
        candidates = second_candidates
    elif replacement == 'window':
        object.__setattr__(first, '_PreparedTreeFrameV3__model',
                           replace(first.model, encoder=TreeEncoderConfigV3(aggregate_window=1)))
    else:
        object.__setattr__(first, '_PreparedTreeFrameV3__cards', second.cards)
    monkeypatch.setattr(np, 'empty', lambda *a, **kw: pytest.fail('binding check must precede row allocation'))
    with pytest.raises(ContractError, match='prepared token/model/cache binding mismatch'):
        next(first.iter_chunks(candidates))


def test_immutable_cache_owners_and_slotted_container_are_admitted_before_allocation(monkeypatch):
    _, _, first, candidates = prepared(inputs([('bronze-club', None)]), [{'type': 'pray'}])
    extra = 12345
    retained = PreparedTreeFrameV3.create(first.tokens, first.model, extra_owned_bytes=extra)
    assert retained.resident_bytes == first.resident_bytes + extra
    assert not hasattr(retained, '__dict__')
    assert isinstance(retained.context.base, bytes) and isinstance(retained.cards.base, bytes)
    for array in (retained.context, retained.cards):
        with pytest.raises(ValueError):
            array.flags.writeable = True
        with pytest.raises(ValueError):
            array.flat[0] = 0
    minimum = (_owned((retained.tokens, retained.model)) + retained.context.nbytes + retained.cards.nbytes
               + sys.getsizeof(retained) + sys.getsizeof(retained.context) + sys.getsizeof(retained.cards))
    assert _owned(retained) >= minimum
    assert retained.resident_bytes >= _owned(retained) + extra
    assert rows(retained, candidates).tobytes() == rows(first, candidates).tobytes()
    # A tokens/model-only budget misses the cache/container ownership and must refuse.
    native_bytes = 1024**2
    old_total = resource_estimate(first.model, model_bytes=native_bytes,
                                 extra_bytes=_owned((first.tokens, first.model)))['total_bytes']
    bounded = replace(first.model, resources=TreeResourcesV2(max_owned_bytes=old_total + 1024))
    monkeypatch.setattr(np, 'zeros', lambda *a, **kw: pytest.fail('sealed owner admission must precede caches'))
    with pytest.raises(ContractError, match='resource admission'):
        PreparedTreeFrameV3.create(first.tokens, bounded, model_bytes=native_bytes)


@pytest.mark.parametrize('cache', ['context', 'cards'])
@pytest.mark.parametrize('metadata', ['dtype', 'shape'])
def test_public_cache_metadata_mutation_cannot_emit_incorrect_rows(monkeypatch, cache, metadata):
    _, _, preparation, candidates = prepared(inputs([('bronze-club', None)]), [{'type': 'pray'}]*3)
    actual = rows(preparation, candidates)
    assert actual[0].tobytes() == reference(preparation.tokens, candidates, 0, 18).tobytes()
    chunks = preparation.iter_chunks(candidates)
    assert next(chunks)[1].tobytes() == actual[:2].tobytes()
    array = getattr(preparation, cache)
    if metadata == 'dtype':
        array.dtype = np.int32
    else:
        array.shape = (3, 543) if cache == 'context' else (64, 18)
    monkeypatch.setattr(np, 'empty', lambda *a, **kw: pytest.fail('cache metadata must reject before rows'))
    with pytest.raises(ContractError, match='prepared token/model/cache binding mismatch'):
        next(chunks)
