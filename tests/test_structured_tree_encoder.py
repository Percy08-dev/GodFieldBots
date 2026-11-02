"""Independent deterministic Tree2 row reference; no learned-encoder oracle."""
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError
from gf_common.structured import CandidateTable
from gf_common.structured_tokens import CARD_CATEGORIES, signature, tokenize_frame, tokenize_table
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV2,
    TreeEncoderConfigV2,
    TreeInferenceConfigV2,
    TreeResourcesV2,
    architecture_identity,
    configured_tree_v2,
    resource_estimate,
)
from gf_models.structured_tree_encoder import PreparedTreeFrame, feature_layout
from tests.test_structured_records import frame_for


def tokens(make_input, *, positions=18, window=128):
    frame, bank = frame_for(make_input(hand=('bronze-club', 'bronze-club'), learned=('fireball', 'aura')))
    actions = [{'type': 'pray'},
        *[{'type': 'play_sequence', 'ordered_elements': [frame.references[slot]], 'target': 2} for slot in (0, 1)],
        {'type': 'play_sequence', 'ordered_elements': [frame.references[3], frame.references[0], frame.references[2]], 'target': None}]
    table = CandidateTable.from_actions(frame, actions, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    return tokenize_frame(frame, bank.view(), aggregate_window=window), tokenize_table(table, frame, positions, aggregate_window=window)


def scalar_log(value):
    number = np.float32(value)
    return np.sign(number) * np.log1p(np.abs(number))


def independent_row(frame, candidates, index, positions):
    # Handwritten field flattening directly follows the accepted row recipe;
    # no production layout, width, transform, card packing or gather helpers.
    def numeric(values):
        return [scalar_log(value) for value in values]
    def card(slot):
        return [*numeric(frame.card_numerics[slot]), *frame.card_present[slot],
                *frame.card_categories[slot], frame.card_mask[slot]]
    chosen = candidates.slots[candidates.offsets[index]:candidates.offsets[index+1]]
    context = [*numeric(frame.state_numerics), *frame.state_present, *frame.state_categories,
        *frame.state_category_present, *[value for slot in range(18) for value in card(slot)],
        *frame.opponent_learned, *numeric(frame.history)]
    candidate = [candidates.kinds[index], candidates.targets[index], *numeric(candidates.numerics[index]),
        *candidates.present[index], scalar_log(len(chosen)), *numeric([chosen.count(slot) for slot in range(18)]),
        *[value for slot in chosen for value in card(slot)], *([0] * (32 * (positions-len(chosen))))]
    return np.asarray([*context, *candidate], dtype=np.float32)


@pytest.mark.parametrize('positions', [18, 128])
@pytest.mark.parametrize('window', [1, 128, 1024])
def test_independent_all_fields_categories_copy_identity_and_chunk_permutation(make_input, positions, window):
    frame, candidates = tokens(make_input, positions=positions, window=window)
    model = ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(max_positions=positions, aggregate_window=window),
        inference=TreeInferenceConfigV2(candidate_chunk=2))
    prepared = PreparedTreeFrame.create(frame, model)
    blocks = list(prepared.iter_chunks(candidates))
    assert [offset for offset, _ in blocks] == [0, 2]
    actual = np.concatenate([block for _, block in blocks])
    assert actual.shape == (4, 1071 + 32 * positions)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    expected = np.stack([independent_row(frame, candidates, index, positions) for index in range(4)])
    np.testing.assert_array_equal(actual, expected)
    layout = feature_layout(model.encoder)
    assert len(layout['feature_names']) == actual.shape[1]
    assert len(layout['categorical_indices']) == 156 + 7 * positions
    categories = actual[:, layout['categorical_indices']]
    np.testing.assert_array_equal(categories.astype(np.int64).astype(np.float32), categories)
    assert np.all(actual[0, -positions*32:] == 0)
    # Same-definition distinct physical copies have identical card values; only
    # the two observed slot-usage columns distinguish the selected own copy.
    differences = np.flatnonzero(actual[1] != actual[2])
    assert differences.tolist() == [layout['feature_names'].index('candidate.used_slot.0'),
                                     layout['feature_names'].index('candidate.used_slot.1')]
    permutation = (3, 0, 2, 1)
    chosen = [candidates.slots[candidates.offsets[i]:candidates.offsets[i+1]] for i in permutation]
    offsets = [0]
    for row in chosen:
        offsets.append(offsets[-1] + len(row))
    permuted = replace(candidates, kinds=tuple(candidates.kinds[i] for i in permutation),
        targets=tuple(candidates.targets[i] for i in permutation), numerics=tuple(candidates.numerics[i] for i in permutation),
        present=tuple(candidates.present[i] for i in permutation), offsets=tuple(offsets), slots=tuple(v for row in chosen for v in row))
    reordered = np.concatenate([block for _, block in prepared.iter_chunks(permuted)])
    np.testing.assert_array_equal(reordered, actual[list(permutation)])


@pytest.mark.parametrize('positions', [18, 128])
def test_synthetic_repeated_capacity_preserves_order_counts_and_empty_rejection(make_input, positions, monkeypatch):
    frame, candidates = tokens(make_input, positions=positions)
    long = replace(candidates, kinds=(candidates.kinds[1],), targets=(candidates.targets[1],),
        numerics=(candidates.numerics[1],), present=(candidates.present[1],), offsets=(0, positions), slots=(2,) * positions)
    model = ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(max_positions=positions))
    prepared = PreparedTreeFrame.create(frame, model)
    actual = next(prepared.iter_chunks(long))[1][0]
    np.testing.assert_array_equal(actual, independent_row(frame, long, 0, positions))
    monkeypatch.setattr(np, 'empty', lambda *args, **kwargs: pytest.fail('empty must reject before row allocation'))
    with pytest.raises(ContractError):
        next(prepared.iter_chunks(replace(long, kinds=(), targets=(), numerics=(), present=(), offsets=(0,), slots=())))


def test_config_closed_roundtrip_architecture_resource_early_refusal_and_optional_free(tmp_path, make_input, monkeypatch):
    model = ConfiguredTreeModelV2()
    assert configured_tree_v2(asdict(model)) == model
    other = replace(model, checkpoint='other', inference=TreeInferenceConfigV2(candidate_chunk=1),
        resources=TreeResourcesV2(max_owned_bytes=1024**3))
    assert architecture_identity(model.architecture()) == architecture_identity(other.architecture())
    assert resource_estimate(model)['array_bytes'] == 256 * 1647 * 4
    for mutation in ({'hand': {}}, {'cuda_visible_devices': None}):
        with pytest.raises(ContractError):
            configured_tree_v2({**asdict(model), **mutation})
    with pytest.raises(ContractError):
        TreeInferenceConfigV2(device='cuda:0')
    with pytest.raises(ContractError):
        TreeInferenceConfigV2(candidate_chunk=True)
    with pytest.raises(ContractError):
        replace(model, inference=TreeInferenceConfigV2(candidate_chunk=65536))
    with pytest.raises(ContractError):
        replace(model, resources=TreeResourcesV2(max_owned_bytes=1))
    frame, _candidates = tokens(make_input)
    monkeypatch.setattr(np, 'zeros', lambda *a, **k: pytest.fail('must refuse ownership before allocation'))
    with pytest.raises(ContractError):
        PreparedTreeFrame.create(replace(frame, recipe_signature=signature(128, aggregate_recipe='historical-structured-62')), model)
    code = "import sys; from gf_models.structured_tree_config import ConfiguredTreeModelV2; ConfiguredTreeModelV2(); assert not ({'numpy','torch','lightgbm'} & sys.modules.keys())"
    subprocess.run([sys.executable, '-c', code], check=True)


def test_pad_missing_and_usability_categories_remain_exact(make_input):
    frame, candidates = tokens(make_input)
    prepared = PreparedTreeFrame.create(frame, ConfiguredTreeModelV2())
    row = next(prepared.iter_chunks(candidates))[1][0]
    names = feature_layout(prepared.model.encoder)['feature_names']
    assert row[names.index('own.0.category.definition')] == frame.card_categories[0][0]
    assert row[names.index('own.0.category.ability')] == 1  # observed missing
    assert row[names.index('own.17.category.ability')] == 0  # absent padded card
    assert row[names.index('own.2.numeric.learned_order')] == scalar_log(1)
    assert row[names.index('own.3.numeric.learned_order')] == scalar_log(2)
    assert row[names.index('own.0.numeric.usable')] == scalar_log(frame.card_numerics[0][7])
    malformed = replace(frame, card_categories=((len(CARD_CATEGORIES[0]) + 2, *frame.card_categories[0][1:]), *frame.card_categories[1:]))
    with pytest.raises(ContractError):
        PreparedTreeFrame.create(malformed, prepared.model)


def test_actual_model_residency_is_carried_through_context_and_chunk_admission(make_input, monkeypatch):
    from gf_models.structured_tree_encoder import _owned
    frame, candidates = tokens(make_input)
    original = ConfiguredTreeModelV2()
    native_bytes = 1024**2
    actual = resource_estimate(original, model_bytes=native_bytes, extra_bytes=_owned((frame, original)))['total_bytes']
    bounded = replace(original, resources=TreeResourcesV2(max_owned_bytes=actual + _owned(candidates)//2))
    prepared = PreparedTreeFrame.create(frame, bounded, model_bytes=native_bytes)
    assert prepared.model_bytes == native_bytes
    monkeypatch.setattr(np, 'empty', lambda *a, **k: pytest.fail('must refuse candidate allocation before actual model+owner admission'))
    with pytest.raises(ContractError, match='resource admission'):
        next(prepared.iter_chunks(candidates))
    tiny = replace(original, resources=TreeResourcesV2(max_owned_bytes=resource_estimate(original)['total_bytes'] + 1024))
    monkeypatch.setattr(np, 'zeros', lambda *a, **k: pytest.fail('must refuse context before actual model admission'))
    with pytest.raises(ContractError, match='resource admission'):
        PreparedTreeFrame.create(frame, tiny, model_bytes=native_bytes)
