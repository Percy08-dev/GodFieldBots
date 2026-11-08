"""Structural final64 packet tests; never a formal parent/Source admission."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import frame_dict
from gf_models.structured_binding import input_binding_game_v3
from gf_training.card_instance_journal import read_card_instance_journal
from gf_training.native_card_instance_final_validation import _decode_final_validation_packet
from gf_training.structured_rollout import (
    CANDIDATE_FIELDS,
    EVENT_FIELDS,
    FRAME_FIELDS,
    _episode_correspondence_digest,
    _token_dtype,
    game_episode_records,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope='module')
def originals(tmp_path_factory):
    fixture = json.loads((ROOT / 'native/tests/fixtures/card_instance_final64_originals.json').read_bytes())
    root = tmp_path_factory.mktemp('UNADMITTED-final64-decoder-originals')
    for name, raw in fixture['files'].items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes.fromhex(raw))
    manifest = json.loads((root / 'manifest.json').read_bytes())
    actions = action_config_v3(fixture['state']['effective_config']['actions'])
    rules = card_instance_rules()
    summary = {'scope': 'UNADMITTED-decoder-structural-evidence-NOT-Source',
        'episodes': [{'episode_index': i, 'episode_id': row['episode_id'], 'viewer': row['viewer'],
            'decisions': len(row['tokens']), 'episode_digest': row['digest'], 'outcome': row['outcome']}
            for i, row in enumerate(fixture['oracle'])]}
    return root, fixture, manifest, actions, rules, summary


def packet_from_reference(originals, index):
    root, fixture, _, actions, rules, summary = originals
    oracle = fixture['oracle'][index]
    path = root / oracle['episode_id'] / f"player-{oracle['viewer']}.jsonl"
    data = read_card_instance_journal(path, actions=actions)
    selected = [data.decisions[i] for i in data.accepted]
    record = game_episode_records(data.bank, tuple(row.frame for row in selected),
        tuple(row.generated.table for row in selected), tuple(row.selection for row in selected),
        'natural', oracle['outcome'], actions=actions, rules=rules)
    arrays = {}
    bank = {'actor': record.bank.actor, 'episode': record.bank.episode,
        'initial_prefix_hash': record.bank._prefixes[0], 'event_records': [],
        'terminal_prefix_hash': record.bank.boundary.prefix_hash}
    for i, raw in enumerate(record.bank._events):
        key = f'dto.event.{i}'
        arrays[key] = np.frombuffer(raw, dtype=np.uint8)
        bank['event_records'].append({'array_key': key, 'prefix_hash': record.bank._prefixes[i+1]})
    decisions = []
    for i, row in enumerate(selected):
        source = next(value for value in data.records if value.get('kind') == 'decision' and value['decision'] == data.accepted[i])
        descriptor = source['table']
        arrays[f'dto.frame.{i}'] = np.frombuffer(canonical(frame_dict(row.frame, layout=LAYOUT)).encode(), dtype=np.uint8)
        arrays[f'dto.columns.{i}'] = np.frombuffer((path.parent / descriptor['file']).read_bytes(), dtype=np.uint8)
        arrays[f'e.0.f.{i}.payload'] = np.frombuffer(row.frame.payload, dtype=np.uint8)
        decisions.append({'frame_array_key': f'dto.frame.{i}', 'candidate_descriptor': descriptor,
            'candidate_array_key': f'dto.columns.{i}', 'selection': asdict(row.selection)})
        tokens = oracle['tokens'][i]
        for fields, prefix, values in ((FRAME_FIELDS, f'e.0.f.{i}', tokens['frame']),
                (EVENT_FIELDS, f'e.0.event-tokens.{i}', tokens['frame']),
                (CANDIDATE_FIELDS, f'e.0.f.{i}.candidate', tokens['candidate'])):
            for field in fields:
                value = np.asarray(values[field], dtype=_token_dtype(field))
                arrays[prefix+'.'+field] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    import hashlib
    frames = [{'frame_hash': frame.frame_hash, 'payload_sha256': hashlib.sha256(frame.payload).hexdigest(),
        'revision': frame.revision, 'interval_start': frame.interval_start,
        'interval_start_hash': frame.interval_start_hash, 'boundary': asdict(frame.boundary),
        'table_hash': table.table_hash, 'selection': asdict(selection)}
        for frame, table, selection in zip(record.frames, record.tables, record.selections, strict=True)]
    metadata = {'schema_version': 1, 'kind': 'card-instance-final-validation-episode-payload',
        'summary_sha256': digest(summary), 'episode_index': index, 'episode_id': oracle['episode_id'],
        'viewer': oracle['viewer'], 'episode_digest': oracle['digest'],
        'binding': input_binding_game_v3(actions.action_contract, rules),
        'endpoint': {'end_kind': 'natural', 'outcome': record.outcome, 'reason': None},
        'bank': bank, 'decisions': decisions, 'correspondence': {'episode_digest': oracle['digest'], 'frames': frames}}
    return pack(metadata, arrays), record


def pack(metadata, arrays):
    metadata = copy.deepcopy(metadata)
    groups = {dtype: [] for dtype in ('uint8', 'int64', 'float32')}
    metadata['arrays'] = []
    for key, value in arrays.items():
        group = groups[str(value.dtype)]
        metadata['arrays'].append({'key': key, 'dtype': str(value.dtype), 'shape': list(value.shape),
            'bank_index': len(group), 'element_offset': 0, 'element_count': value.size})
        group.append(np.frombuffer(value.tobytes(), dtype=value.dtype))
    return canonical(metadata).encode(), *(tuple(group) for group in groups.values())


def decode(originals, packet, index, **overrides):
    _, _, _, actions, rules, summary = originals
    kwargs = {'summary': summary, 'episode_index': index, 'actions': actions, 'rules': rules,
        'aggregate_window': 2, 'max_owned_bytes': 1024**3, 'retained_owned_bytes': sum(bank.nbytes for group in packet[1:] for bank in group),
        'max_record_bytes': 8*1024**2, 'reserve': lambda _: None, 'check': lambda: None, **overrides}
    return _decode_final_validation_packet(packet, **kwargs)


@pytest.mark.parametrize('index', range(64))
def test_all64_original_packet_dtos_raw_tokens_and_zero_endpoints_match_full_python_reader(originals, index):
    before = {name: value for name, value in sys.modules.items() if name == 'torch' or name.startswith('torch.')}
    packet, expected = packet_from_reference(originals, index)
    result = decode(originals, packet, index)
    assert result == expected
    assert _episode_correspondence_digest(result) == _episode_correspondence_digest(expected)
    assert before == {name: value for name, value in sys.modules.items() if name == 'torch' or name.startswith('torch.')}


@pytest.mark.parametrize('mutation', ['kind', 'viewer', 'order', 'selected', 'prefix', 'raw_token', 'foreign_array',
    'shape', 'writable', 'old_binding', 'wrong_digest', 'raw_frame', 'codec', 'full_precision_selection', 'raw_value'])
def test_malformed_mixed_wrong_viewer_future_and_foreign_packets_refuse(originals, mutation):
    packet, _ = packet_from_reference(originals, 1)
    metadata = json.loads(packet[0])
    if mutation == 'raw_value':
        item = next(v for v in metadata['arrays'] if v['key'] == 'e.0.f.0.card_mask')
        groups = [list(group) for group in packet[1:]]
        value = groups[0][item['bank_index']].copy()
        value[item['element_offset']] = 2
        groups[0][item['bank_index']] = np.frombuffer(value.tobytes(), dtype=value.dtype)
        packet = packet[0], *(tuple(group) for group in groups)
    elif mutation == 'writable':
        groups = [list(group) for group in packet[1:]]
        groups[0][0] = groups[0][0].copy()
        packet = packet[0], *(tuple(group) for group in groups)
    else:
        if mutation == 'kind': metadata['kind'] = 'card-instance-recorded-cohort-payload'
        elif mutation == 'viewer': metadata['viewer'] = 1
        elif mutation == 'order': metadata['episode_index'] = 0
        elif mutation == 'selected': metadata['decisions'][0]['selection']['index'] = 1000
        elif mutation == 'prefix': metadata['bank']['terminal_prefix_hash'] = '0'*64
        elif mutation == 'raw_token':
            item = next(v for v in metadata['arrays'] if v['key'] == 'e.0.f.0.state_numerics')
            item['element_count'] += 1
        elif mutation == 'foreign_array': metadata['arrays'][0]['key'] = 'foreign'
        elif mutation == 'shape': metadata['arrays'][-1]['shape'] = [10000]
        elif mutation == 'old_binding': metadata['binding']['purpose'] = 'component'
        elif mutation == 'wrong_digest': metadata['episode_digest'] = '0'*64
        elif mutation == 'raw_frame': metadata['decisions'][0]['frame_array_key'] = 'e.0.f.0.payload'
        elif mutation == 'codec': metadata['decisions'][0]['candidate_descriptor']['schema_version'] = 1
        elif mutation == 'full_precision_selection': metadata['decisions'][0]['selection']['value'] = .987654
        packet = canonical(metadata).encode(), *packet[1:]
    with pytest.raises((ContractError, ValueError, IndexError)):
        decode(originals, packet, 1)


def test_joint_budget_refuses_before_any_json_or_column_decode(originals, monkeypatch):
    import gf_training.native_card_instance_final_validation as module
    packet, _ = packet_from_reference(originals, 1)
    called = []
    monkeypatch.setattr(module, '_json', lambda *_: called.append('json'))
    with pytest.raises(ContractError, match='BEFORE decode'):
        decode(originals, packet, 1, max_owned_bytes=100, retained_owned_bytes=99)
    assert not called


def test_column_growth_is_admitted_before_codec_decode_and_cancel_propagates(originals, monkeypatch):
    import gf_training.native_card_instance_final_validation as module
    packet, _ = packet_from_reference(originals, 1)
    calls = []
    original = module.decode_candidate_columns_v2
    def codec(*args, **kwargs):
        calls.append('codec')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'decode_candidate_columns_v2', codec)
    class Stop(Exception): pass
    def reserve(n):
        # Metadata admission succeeds; fail at subsequent frame/column growth.
        if n > len(packet[0])*64 + 65536 + 1024*(len(json.loads(packet[0])['arrays'])+1) + 1024:
            raise Stop('controlled caller reservation failure')
    with pytest.raises(Stop):
        decode(originals, packet, 1, reserve=reserve)
    assert not calls
    with pytest.raises(Stop):
        decode(originals, packet, 1, check=lambda: (_ for _ in ()).throw(Stop('cancel')))


def test_controlled_population_crosses_both_viewers_outcomes_and_zero_decisions(originals):
    rows = originals[1]['oracle']
    for viewer in (1, 2):
        for result in ('draw', 'win', 'loss'):
            for empty in (False, True):
                assert any(row['viewer'] == viewer and (not row['tokens']) == empty and
                    ('draw' if row['outcome'] == 'draw' else 'win' if row['outcome']['win']['winner'] == viewer else 'loss') == result
                    for row in rows)


@pytest.mark.parametrize('index', [0, 1, 2, 3, 5, 7])
def test_raw_arrays_persist_reload_without_tokens_or_original_dto_changes(originals, index, tmp_path):
    from gf_models.sharded_arrays import read_bundle, write_bundle
    packet, expected = packet_from_reference(originals, index)
    metadata = json.loads(packet[0])
    banks = dict(zip(('uint8','int64','float32'), packet[1:], strict=True))
    arrays = {item['key']: banks[item['dtype']][item['bank_index']][
        item['element_offset']:item['element_offset']+item['element_count']].reshape(item['shape'])
        for item in metadata['arrays']}
    if not arrays:
        assert not expected.frames and not tuple(tmp_path.iterdir())
        return  # Real zero endpoint; no fabricated array or persistence sentinel.
    bundle = write_bundle(tmp_path/'raw', 'raw', arrays, shard_bytes=65536)
    reference = write_bundle(tmp_path/'reference', 'raw', arrays, shard_bytes=65536)
    assert bundle == reference
    for name in bundle['files']:
        assert (tmp_path/'raw'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()
    reloaded = read_bundle(tmp_path/'raw', bundle)
    for key, value in arrays.items():
        assert value.dtype == reloaded[key].dtype and value.shape == reloaded[key].shape
        assert value.tobytes() == reloaded[key].tobytes()
    assert decode(originals, pack(metadata, reloaded), index) == expected


@pytest.mark.parametrize('index', range(64))
def test_opt_in_same_rust_exported_packet_decodes_to_identical_original(originals, index):
    directory = os.environ.get('GFB_FINAL64_TEST_EXPORT_DIR')
    if directory is None:
        pytest.skip('opt-in SAME native unit-export packet directory required')
    root = Path(directory)
    summary = json.loads((root/'summary.json').read_bytes())
    dump = json.loads((root/f'episode-{index:02}.json').read_bytes())
    metadata = dump['metadata']
    arrays = {item['key']: np.frombuffer(bytes.fromhex(dump['array_hex'][item['key']]), dtype=item['dtype']).reshape(item['shape']) for item in metadata['arrays']}
    packet = pack(metadata, arrays)
    result = decode(originals, packet, index, summary=summary)
    _, expected = packet_from_reference(originals, index)
    assert result == expected


def test_fresh_decoder_import_requires_no_optional_model_stack():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','lightgbm'): raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import gf_training.native_card_instance_final_validation
assert 'torch' not in sys.modules and 'lightgbm' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True, env={**os.environ, 'PYTHONPATH': str(ROOT/'src')}, timeout=30)


def test_private_candidate_additive_four_methods_and_closed_capabilities_before_io(tmp_path):
    import inspect
    import time
    native = pytest.importorskip('gf_native')
    cls = native.CardInstanceNativeSession
    if not hasattr(cls, 'inspect_final_validation'):
        pytest.skip('new isolated final64 diagnostic candidate required; installed ELF unchanged')
    fields = {
        'inspect_final_validation': ('self','request','retained_python_bytes'),
        'export_final_validation': ('self','episode_index','retained_python_bytes'),
        'recheck_final_validation': ('self','retained_python_bytes'),
        'close_final_validation': ('self',),
    }
    for name, keys in fields.items():
        assert callable(getattr(cls, name))
        assert tuple(inspect.signature(getattr(cls, name)).parameters) == keys
    info = dict(native.build_info())
    assert info['card_instance_contract'] == 'participant-structured-v2;journal6;rollout4;generation3;model3'
    assert info['card_instance_signature'] == 'ed0c61152b78fa984aad342be22c6b9d6ef49b12b8f0d1c0f0de89724468002f'
    assert info['card_instance_models'] == 'mlp,gru,transformer'
    from gf_common.preprocessing import PreparationConfig
    from gf_training.card_instance_formal import require_native_formal_controls
    from gf_training.native_card_instance import _require_final_validation_native
    assert info['card_instance_final_validation'] == 'collection5-final64;journal6;codec2'
    assert info['card_instance_formal_controls'] == 'run3-formal-parent-v1;declaration1;stop1'
    assert info['card_instance_fixture_tokens'] == 'component-envelope1;participant-structured-v2;codec2'
    config = PreparationConfig(backend='rust', reader='participant-structured-v2')
    assert _require_final_validation_native(config, native) is native
    require_native_formal_controls(native)
    settings = canonical({'memory_bytes': 128*1024**2, 'scratch_bytes': 64*1024**2,
        'max_source_file_bytes': 64*1024**2, 'max_record_bytes': 8*1024**2, 'chunk_target_bytes': 1024**2}).encode()
    with pytest.raises((ValueError, RuntimeError)):
        cls(b'{}', settings, time.monotonic()+30)
    assert list(tmp_path.iterdir()) == []
