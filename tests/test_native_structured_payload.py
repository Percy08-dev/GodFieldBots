"""Private C1 payload seam and exact Source digests; no native/source authority."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, canonical
from gf_models.arrays import sha256
from gf_models.sharded_arrays import read_bundle, write_owned_json
from gf_models.structured_sealed import SEMANTIC_KEYS
from gf_training.structured_rollout import (
    _episode_correspondence_digest,
    _owned_bytes,
    _read_recorded_rollout,
    _source_correspondence,
    _write_prepared_payload,
)
from tests.test_structured_ppo_source import fixture, source


def metadata(manifest):
    return {key: value for key, value in manifest.items() if key not in ('schema_version', 'kind', 'data')}


def read_recorded(directory, model, declaration, correspondence=None, records=None):
    return _read_recorded_rollout(directory, sha256(directory / 'rollout.json'), model,
        action_contract=engineering_contract(), declaration=declaration, expected_records=records,
        expected_correspondence=correspondence, max_owned_bytes=512 * 1024**2, extra_owned_bytes=0,
        check=lambda: None, reserve=lambda _: None)


def test_full_original16_role_payload_matches_public_writer_without_record_copy_or_tokenization(tmp_path, make_input, monkeypatch):
    import gf_training.structured_rollout as module
    values = fixture(tmp_path, make_input)
    model, declaration = values[2], values[4]
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'reference')
        arrays = read_bundle(tmp_path / 'reference', manifest['data'])
        meta = metadata(manifest)
        owned = _owned_bytes((meta, arrays, model))
        correspondence = _source_correspondence(original.proof['source_set_sha256'], declaration,
            tuple(_episode_correspondence_digest(record) for record in original.records))
        def forbidden(*args, **kwargs):
            pytest.fail('prepared publisher reconstructed/tokenized Source')
        with monkeypatch.context() as scoped:
            scoped.setattr(module, 'tokenize_frame', forbidden)
            scoped.setattr(module, 'tokenize_table', forbidden)
            scoped.setattr(module, 'DecisionFrame', forbidden)
            written = _write_prepared_payload(tmp_path / 'prepared', meta, arrays, model,
                resident_bytes=owned, extra_owned_bytes=original._snapshot.resident_bytes,
                max_owned_bytes=512 * 1024**2, check=original._snapshot.check, reserve=lambda _: None)
        assert canonical(written) == canonical(manifest)
        for name in ('rollout.json', *manifest['data']['files']):
            assert (tmp_path / 'prepared' / name).read_bytes() == (tmp_path / 'reference' / name).read_bytes()
        loaded = read_recorded(tmp_path / 'prepared', model, declaration, correspondence)
        assert len(loaded) == len(loaded.episodes) == 16
        for record, episode in zip(original.records, loaded.episodes, strict=True):
            assert _episode_correspondence_digest(episode.records) == _episode_correspondence_digest(record)
        original._snapshot.recheck()


@pytest.mark.parametrize('mutation', ['extra_array', 'writeable', 'shape', 'dtype', 'wrong_count', 'unknown_metadata', 'scalar', 'fp32_overflow'])
def test_prepared_payload_rejects_before_publication(tmp_path, make_input, mutation):
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'reference')
        arrays = read_bundle(tmp_path / 'reference', manifest['data'])
        meta = copy.deepcopy(metadata(manifest))
        key = 'e.0.f.0.history'
        if mutation == 'extra_array':
            arrays['unowned'] = np.empty(0, np.uint8)
        elif mutation == 'writeable':
            arrays[key] = arrays[key].copy()
        elif mutation == 'shape':
            arrays[key] = arrays[key][:-1]
        elif mutation == 'dtype':
            arrays[key] = arrays[key].astype(np.float32)
            arrays[key].flags.writeable = False
        elif mutation == 'wrong_count':
            meta['decision_count'] += 1
        elif mutation == 'unknown_metadata':
            meta['unknown'] = True
        elif mutation == 'scalar':
            arrays['e.0.old_value'] = np.asarray([-0.0], np.float32)
            arrays['e.0.old_value'].flags.writeable = False
        else:
            meta['episodes'][0]['selections'][0]['value'] = 1e308
        with pytest.raises(ContractError):
            _write_prepared_payload(tmp_path / 'blocked', meta, arrays, values[2],
                resident_bytes=_owned_bytes((meta, arrays, values[2])), extra_owned_bytes=original._snapshot.resident_bytes,
                max_owned_bytes=512 * 1024**2, check=original._snapshot.check, reserve=lambda _: None)
        assert not (tmp_path / 'blocked').exists()


def test_full_selection_precision_changes_digest_even_when_fp32_numeric_is_selfconsistent(tmp_path, make_input):
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'numeric')
        declaration = {key: manifest[key] for key in SEMANTIC_KEYS}
        expected = _source_correspondence(original.proof['source_set_sha256'], declaration,
            tuple(_episode_correspondence_digest(record) for record in original.records))
        selection = manifest['episodes'][0]['selections'][0]
        old = selection['value']
        selection['value'] = float(np.nextafter(old, np.inf))
        assert selection['value'] != old and np.float32(selection['value']) == np.float32(old)
        write_owned_json(tmp_path / 'altered.json', manifest)
        # This is a deliberately resealed local attack fixture, not publication.
        (tmp_path / 'numeric/rollout.json').write_text(canonical(manifest) + '\n')
        selfconsistent = read_recorded(tmp_path / 'numeric', values[2], declaration)
        assert len(selfconsistent) == 16
        with pytest.raises(ContractError, match='ORIGINAL native Source'):
            read_recorded(tmp_path / 'numeric', values[2], declaration, expected)


@pytest.mark.parametrize('mutation', ['count', 'order', 'semantic', 'binding', 'exclusive'])
def test_expected_digest_count_order_semantics_and_mode_are_strict(tmp_path, make_input, mutation):
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        original.write_numeric(tmp_path / 'numeric')
        declaration = values[4]
        digests = tuple(_episode_correspondence_digest(record) for record in original.records)
        if mutation == 'count':
            digests = digests[:-1]
        elif mutation == 'order':
            digests = digests[::-1]
        expected = _source_correspondence(original.proof['source_set_sha256'], declaration, digests)
        if mutation == 'semantic':
            expected['semantics_sha256'] = '0' * 64
        elif mutation == 'binding':
            expected['binding_sha256'] = '0' * 64
        with pytest.raises(ContractError):
            read_recorded(tmp_path / 'numeric', values[2], declaration, expected,
                records=original.records if mutation == 'exclusive' else None)


def test_correspondence_includes_zero_and_interrupted_final_bank_and_exact_closure(make_input):
    from gf_common.structured import EventBank, EventDelta
    from gf_training.structured_rollout import EpisodeRecords
    bank = EventBank(1, 'zero-original')
    def record(reason):
        return EpisodeRecords(bank.view(), (), (), (), 'cancelled', None, reason,
            action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    first = _episode_correspondence_digest(record('exact cancellation'))
    assert first != _episode_correspondence_digest(record('different cancellation'))
    bank.append(EventDelta(1, bank.episode, 0, bank.boundary.prefix_hash,
        [{'type': 'action_accepted', 'player': 1}]))
    assert first != _episode_correspondence_digest(record('exact cancellation'))


def test_prepared_first_raw_blob_budget_rejects_before_any_finite_scan(tmp_path, make_input, monkeypatch):
    import gf_training.structured_rollout as module
    values = fixture(tmp_path, make_input)
    with source(values) as original:
        manifest = original.write_numeric(tmp_path / 'reference')
        arrays = read_bundle(tmp_path / 'reference', manifest['data'])
        meta = copy.deepcopy(metadata(manifest))
        episode = meta['episodes'][0]
        episode['boundary'] = {'end': 1, 'prefix_hash': 'a' * 64}
        episode['raw_blocks'] = [{'start': 0, 'end': 1, 'key': 'e.0.raw-events.0'}]
        large = np.zeros(2 * 1024**2, np.uint8)
        large.flags.writeable = False
        arrays['e.0.raw-events.0'] = large
        def forbidden(*args, **kwargs):
            pytest.fail('prospective raw bytes were scanned before joint admission')
        monkeypatch.setattr(module.np, 'isfinite', forbidden)
        with pytest.raises(ContractError, match='reservation'):
            _write_prepared_payload(tmp_path / 'blocked', meta, arrays, values[2], resident_bytes=0,
                extra_owned_bytes=0, max_owned_bytes=1024**2, check=lambda: None, reserve=lambda _: None)
        assert not (tmp_path / 'blocked').exists()
