"""Pure component diagnostic FFI; never a Source, game or fitting permission."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gf_common.domain import canonical, digest
from gf_models.sharded_arrays import read_bundle, write_bundle

ROOT = Path(__file__).parents[1]
ORACLE = ROOT / 'native/tests/fixtures/card_instance_component_tokens_python_rules_v2.json'
CASES = json.loads(ORACLE.read_text())['cases']


@pytest.fixture
def native():
    module = pytest.importorskip('gf_native')
    if not hasattr(module, 'CardInstanceFixtureTokens'):
        pytest.skip('new isolated diagnostic candidate required; installed extension unchanged')
    return module


def settings(memory=128*1024**2, **override):
    return canonical({'memory_bytes': memory, 'scratch_bytes': 64*1024**2,
        'max_source_file_bytes': 64*1024**2, 'max_record_bytes': 64*1024**2,
        'chunk_target_bytes': 65536, **override}).encode()


def construct(native, case=CASES[0], *, raw=None, payloads=None, limits=None, aggregate_window=2, **kwargs):
    return native.CardInstanceFixtureTokens(case['envelope_utf8'].encode() if raw is None else raw,
        tuple(bytes.fromhex(v) for v in case['columns_hex']) if payloads is None else payloads,
        settings() if limits is None else limits, time.monotonic()+60, aggregate_window=aggregate_window, **kwargs)


def unpack(packet):
    assert type(packet) is tuple and len(packet) == 4
    metadata = json.loads(packet[0])
    banks = dict(zip(('uint8','int64','float32'), packet[1:], strict=True))
    arrays = {}
    for dtype, values in banks.items():
        assert type(values) is tuple
        for bank in values:
            assert str(bank.dtype) == dtype and bank.flags.c_contiguous and not bank.flags.writeable
    for descriptor in metadata['arrays']:
        bank = banks[descriptor['dtype']][descriptor['bank_index']]
        start, count = descriptor['element_offset'], descriptor['element_count']
        arrays[descriptor['key']] = bank[start:start+count].reshape(descriptor['shape'])
    return metadata, arrays


@pytest.mark.parametrize('case', CASES, ids=lambda v: v['label'])
def test_component_native_full_tokens_metadata_offsets_order_and_persistence_match_python(native, case, tmp_path):
    torch_before = {name: module for name, module in sys.modules.items()
        if name == 'torch' or name.startswith('torch.')}
    owner = construct(native, case)
    packet = owner.export_tokens()
    metadata, arrays = unpack(packet)
    assert metadata == case['metadata']
    assert set(arrays) == set(case['arrays'])
    assert 'qualified' not in metadata and 'source_proof' not in metadata
    assert metadata['input_binding']['purpose'] == 'component'
    assert metadata['outcome_origin'] == 'declared-component-fixture'
    expected = {}
    for key, oracle in case['arrays'].items():
        array = arrays[key]
        assert str(array.dtype) == oracle['dtype'] and list(array.shape) == oracle['shape']
        assert array.tobytes() == bytes.fromhex(oracle['bytes_hex'])
        expected[key] = np.frombuffer(bytes.fromhex(oracle['bytes_hex']), dtype=oracle['dtype']).reshape(oracle['shape'])
    if arrays:
        actual_bundle = write_bundle(tmp_path/'native', 'raw', arrays, shard_bytes=65536)
        reference_bundle = write_bundle(tmp_path/'python', 'raw', expected, shard_bytes=65536)
        assert canonical(actual_bundle) == canonical(reference_bundle)
        for file in actual_bundle['files']:
            assert (tmp_path/'native'/file).read_bytes() == (tmp_path/'python'/file).read_bytes()
        persisted = read_bundle(tmp_path/'native', actual_bundle)
        assert set(persisted) == set(arrays)
        for key in arrays:
            assert persisted[key].tobytes() == arrays[key].tobytes()
    else:
        assert not metadata['decisions'] and all(not banks for banks in packet[1:])
        assert not tuple(tmp_path.iterdir())
    owner.close()
    owner.close()
    with pytest.raises((ValueError, RuntimeError), match='closed|failed'):
        owner.export_tokens()
    torch_after = {name: module for name, module in sys.modules.items()
        if name == 'torch' or name.startswith('torch.')}
    assert torch_after.keys() == torch_before.keys()
    assert all(torch_after[name] is module for name, module in torch_before.items())


@pytest.mark.parametrize('preloaded', [False, True], ids=['clean-no-optional-import', 'preloaded-sentinel'])
def test_component_operation_preserves_optional_import_boundary(native, tmp_path, preloaded):
    # A fresh interpreter provides the absence check without depending on test
    # ordering or installing Torch. The sentinel reproduces an already-loaded
    # module, while the import hook also detects cached-module import attempts.
    code = '''
import builtins,sys,types
from pathlib import Path
assert 'torch' not in sys.modules
sentinel=types.ModuleType('torch') if sys.argv[2]=='True' else None
if sentinel is not None:sys.modules['torch']=sentinel
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','lightgbm','pyarrow'):raise AssertionError('optional import: '+name)
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
import gf_native
from tests import test_card_instance_native_fixture_tokens as helper
helper.test_component_native_full_tokens_metadata_offsets_order_and_persistence_match_python(
 gf_native,helper.CASES[0],Path(sys.argv[1]))
if sentinel is None:assert 'torch' not in sys.modules
else:assert sys.modules['torch'] is sentinel
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), str(preloaded)],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_component_marker_exact_live_preflight_and_no_source_methods(native):
    info = dict(native.build_info())
    from gf_training.native_card_instance_fixture import FIXTURE_CONTRACT, _preflight
    assert info['card_instance_fixture_tokens'] == FIXTURE_CONTRACT
    assert _preflight(native, diagnostic=False)['build'] == info
    assert info['card_instance_contract'] == 'participant-structured-v2;journal6;rollout4;generation3;model3'
    assert info['card_instance_signature'] == 'ed0c61152b78fa984aad342be22c6b9d6ef49b12b8f0d1c0f0de89724468002f'
    assert info['card_instance_models'] == 'mlp,gru,transformer'
    assert info['structured_preprocessing_contract'] == 'participant-structured-v1;journal=5;rollout=3;generation=2;model=2'
    assert info['structured_default_signature'] == '4b4ae064f08192a322de83e24e4a93ef4191c7d73396997d59cd7710cd05cb04'
    for method in ('authorize','begin_update','prepare_commit','bind_numeric','control_summary','finalize_read'):
        assert not hasattr(native.CardInstanceFixtureTokens, method)


@pytest.mark.parametrize('window', [None, 1])
def test_component_default_and_actual_window_use_same_python_recipe(native, window):
    from gf_common.candidate_columns import decode_candidate_columns_v2
    from gf_common.structured import EventBank, event_delta
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.structured_tokens import signature, tokenize_frame, tokenize_table
    from gf_common.structured_wire import parse_frame
    case = CASES[0]
    raw = case['envelope']
    payloads = tuple(bytes.fromhex(value) for value in case['columns_hex'])
    arguments = {} if window is None else {'aggregate_window': window}
    owner = native.CardInstanceFixtureTokens(case['envelope_utf8'].encode(), payloads, settings(),
        time.monotonic()+60, **arguments)
    metadata, arrays = unpack(owner.export_tokens())
    actual_window = 128 if window is None else window
    assert metadata['aggregate_window'] == actual_window
    assert metadata['token_signature'] == signature(actual_window, layout=layout)
    assert metadata['layout_signature'] == signature(128, layout=layout)
    bank = EventBank(raw['viewer'], raw['episode'], layout=layout)
    bank.append(event_delta(bank.actor, bank.episode, 0, bank.boundary.prefix_hash, raw['events'], layout=layout))
    row = raw['decisions'][0]
    frame = parse_frame(row['frame'], layout=layout)
    table = decode_candidate_columns_v2(payloads[0], row['columns'], layout=layout, action_contract=layout.action_contract,
        purpose='component', synthetic_capacity=False, max_positions=18)
    tokens = tokenize_frame(frame, bank.view(), aggregate_window=actual_window, layout=layout)
    candidates = tokenize_table(table, frame, 18, aggregate_window=actual_window, layout=layout)
    for key, array in arrays.items():
        expected = getattr(candidates if '.candidate.' in key else tokens, key.rsplit('.', 1)[1])
        assert array.tobytes() == np.asarray(expected, dtype=array.dtype).tobytes()
    owner.close()


@pytest.mark.parametrize('generation', [1, 2])
@pytest.mark.parametrize('viewer', [1, 2])
def test_component_generation_split_and_selected_roles_are_internally_bound(native, generation, viewer):
    # Opaque declared IDs exercise structural consistency only; no inference
    # artifact or external declaration membership is asserted by this fixture.
    case = next(value for value in CASES if value['envelope']['viewer'] == viewer)
    raw = copy.deepcopy(case['envelope'])
    raw.update(generation=generation, split='validation' if generation == 2 else 'train',
        cohort_id=f'p17-g{generation}', selected_policy_id=digest({'component-generation': generation, 'viewer': viewer}))
    raw['roles'][viewer-1].update(policy_generation=generation, policy_id=raw['selected_policy_id'])
    encoded = canonical(raw).encode()
    owner = construct(native, case, raw=encoded)
    metadata, arrays = unpack(owner.export_tokens())
    expected = copy.deepcopy(case['metadata'])
    for key in ('generation', 'split', 'cohort_id', 'selected_policy_id', 'roles'):
        expected[key] = raw[key]
    expected['envelope_sha256'] = hashlib.sha256(encoded).hexdigest()
    assert metadata == expected
    for key, array in arrays.items():
        assert array.tobytes() == bytes.fromhex(case['arrays'][key]['bytes_hex'])
    owner.close()


@pytest.mark.parametrize('invalid', ['envelope-bytearray','envelope-memoryview','columns-list','column-bytearray','column-memoryview',
    'missing-column','extra-column','record-bound','source-bound','extra-settings','cancelled','zero-window','oversized-window'])
def test_component_immutable_inputs_record_bounds_and_cancel_refuse_before_any_packet(native, invalid):
    case = CASES[0]
    raw = case['envelope_utf8'].encode()
    payloads = tuple(bytes.fromhex(v) for v in case['columns_hex'])
    limits = settings()
    kwargs = {}
    if invalid == 'envelope-bytearray':
        raw = bytearray(raw)
    elif invalid == 'envelope-memoryview':
        raw = memoryview(raw)
    elif invalid == 'columns-list':
        payloads = list(payloads)
    elif invalid == 'column-bytearray':
        payloads = (bytearray(payloads[0]),)
    elif invalid == 'column-memoryview':
        payloads = (memoryview(payloads[0]),)
    elif invalid == 'missing-column':
        payloads = ()
    elif invalid == 'extra-column':
        payloads += payloads
    elif invalid == 'record-bound':
        limits = settings(max_record_bytes=len(raw)-1)
    elif invalid == 'source-bound':
        limits = settings(max_record_bytes=1, max_source_file_bytes=1)
    elif invalid == 'extra-settings':
        limits = settings(qualified=True)
    elif invalid == 'cancelled':
        handle = native.CancelHandle(time.monotonic()+60)
        handle.cancel()
        kwargs['cancel_handle'] = handle
    else:
        kwargs['aggregate_window'] = 0 if invalid == 'zero-window' else 65537
    with pytest.raises((TypeError, ValueError, RuntimeError, MemoryError)):
        construct(native, raw=raw, payloads=payloads, limits=limits, **kwargs)


def test_component_partial_constructor_failure_joins_retirement_before_exception(native):
    task_root = Path('/proc/self/task')
    def retirement_threads():
        result = []
        for path in task_root.glob('*/comm'):
            try:
                value = path.read_text()
            except FileNotFoundError:
                continue
            if value.startswith('gf-native-ret'):
                result.append(path.parent.name)
        return result
    before = retirement_threads()
    case = next(value for value in CASES if len(value['envelope']['decisions']) == 3)
    raw = copy.deepcopy(case['envelope'])
    raw['decisions'][1]['selected_index'] = 999999
    with pytest.raises(ValueError) as failed:
        construct(native, case, raw=canonical(raw).encode())
    assert not hasattr(failed.value, '_native_fixture_cleanup_owner')  # Confirmed close/join path.
    assert retirement_threads() == before


def seal(frame):
    binding = {key:value for key,value in frame.items() if key != 'frame_hash'}
    contract = CASES[0]['metadata']['input_binding']['action_contract']
    frame['frame_hash'] = digest({**binding, 'input_recipe': contract['input_recipe'],
        'policy_input_version': contract['policy_input_version'], 'action_contract': contract})


@pytest.mark.parametrize('mutation', ['game','journal','http','outcome','role','policy','selected','purpose','old-codec','mixed-profile',
    'missing-codebook','foreign-prefix','foreign-viewer','future-tail','foreign-ref','stale-hints','columns-order'])
def test_component_malformed_mixed_roles_order_and_sealed_prefix_refuse(native, mutation):
    case = next(v for v in CASES if len(v['envelope']['decisions']) == 3) if mutation == 'columns-order' else CASES[0]
    raw = copy.deepcopy(case['envelope'])
    payloads = tuple(bytes.fromhex(v) for v in case['columns_hex'])
    row = raw['decisions'][0]
    frame = row['frame']
    if mutation in ('game','journal','http'):
        raw['kind'] = {'game':'card-instance-source-controls','journal':'header','http':'observation'}[mutation]
    elif mutation == 'outcome':
        raw['outcome']['winner'] = 0
    elif mutation == 'role':
        raw['roles'].reverse()
    elif mutation == 'policy':
        raw['selected_policy_id'] = 'f'*64
    elif mutation == 'selected':
        row['selected_index'] = row['columns']['count']
    elif mutation == 'purpose':
        row['columns']['purpose'] = 'qualified_game'
    elif mutation == 'old-codec':
        row['columns']['schema_version'] = 1
    elif mutation == 'mixed-profile':
        row['columns']['action_contract']['profile_id'] = 'old-foreign-profile'
    elif mutation == 'missing-codebook':
        del row['columns']['codebook_sha256']
    elif mutation == 'foreign-prefix':
        frame['boundary']['prefix_hash'] = '0'*64
        seal(frame)
    elif mutation == 'foreign-viewer':
        raw['viewer'] = 2
    elif mutation == 'future-tail':
        raw['events'].append(copy.deepcopy(raw['events'][0]))
    elif mutation == 'foreign-ref':
        frame['payload']['observation']['own']['cards'][0]['card_ref'] = 'server_private_ref'
        seal(frame)
    elif mutation == 'stale-hints':
        frame['payload']['action_options']['state_revision'] += 1
        seal(frame)
    else:
        payloads = tuple(reversed(payloads))
    with pytest.raises((ValueError, RuntimeError, MemoryError)):
        construct(native, case, raw=canonical(raw).encode(), payloads=payloads)


def test_component_arrays_hold_base_owner_after_core_join_close_and_drop(native):
    owner = construct(native)
    packet = owner.export_tokens()
    _, arrays = unpack(packet)
    retained = arrays['e.0.f.0.card_categories']
    original = retained.tobytes()
    assert any(isinstance(value.base, native.IntegerOwner) for value in packet[2])
    owner.close()
    assert owner.used_bytes() > 0  # Only independently exported RAM/metadata leases remain.
    del packet, arrays, owner
    gc.collect()
    assert retained.tobytes() == original and not retained.flags.writeable
    with pytest.raises(ValueError):
        retained.flags.writeable = True


@pytest.mark.parametrize('failure', ['caller-quota','post-retirement-wrapper-quota','cancel'])
def test_component_admitted_failure_cannot_retry_after_repair_and_close_releases_owner(native, failure):
    owner = construct(native, limits=settings(memory=32*1024**2 if failure == 'post-retirement-wrapper-quota' else 128*1024**2))
    if failure == 'cancel':
        owner.cancel_handle().cancel()
    with pytest.raises((MemoryError, RuntimeError, ValueError)):
        owner.export_tokens(12*1024**3 if failure == 'caller-quota' else 0)
    with pytest.raises((RuntimeError, ValueError), match='closed|failed'):
        owner.export_tokens(0)
    owner.close()
    assert owner.used_bytes() == 0
